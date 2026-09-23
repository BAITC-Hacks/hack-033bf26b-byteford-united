"""Generate a human-readable local report for one mock agent run.

This is a development and presentation tool.  It uses the public mock scorer
and never participates in ``Agent.act`` or the judging decision path.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd

from agent import Agent
from mock_environment import make_mock_env, _mock_fallback, _mock_impact_model
from scoring_core import (
    CHANNELS,
    MAX_CAMPAIGNS,
    MAX_CUSTOMERS_PER_CAMPAIGN,
    MAX_TOTAL_CONTACTS,
    TOTAL_BUDGET,
    apply_filters,
    sanitize_campaigns,
    score_campaign,
    score_campaigns,
)


BASE_DIR = Path(__file__).resolve().parent
FILTER_COLUMNS = [
    "filter_arpu_segment",
    "filter_data_segment",
    "filter_call_segment",
    "filter_current_tariff",
]


def _format_number(value: float, digits: int = 0) -> str:
    return f"{float(value):,.{digits}f}".replace(",", " ")


def _progress(label: str, used: float, limit: float, unit: str = "") -> str:
    percent = 0.0 if limit <= 0 else min(max(100.0 * used / limit, 0.0), 100.0)
    return (
        '<div class="progress-block">'
        f'<div class="progress-label"><span>{escape(label)}</span>'
        f'<strong>{_format_number(used)} / {_format_number(limit)} {escape(unit)}</strong></div>'
        f'<div class="progress"><span style="width:{percent:.1f}%"></span></div>'
        f'<div class="progress-note">Использовано {percent:.1f}%</div>'
        "</div>"
    )


def _table(frame: pd.DataFrame, *, classes: str = "data-table") -> str:
    if frame.empty:
        return '<p class="muted">Нет данных.</p>'
    return frame.to_html(index=False, border=0, classes=classes, escape=True)


def _contact_ledger(
    campaigns: pd.DataFrame,
    profile: pd.DataFrame,
    impact_model: pd.DataFrame,
    tariffs: pd.DataFrame,
    n_pilots: int,
) -> pd.DataFrame:
    """Reproduce public contact allocation and return one row per contact."""

    fallback_conversion = float(impact_model["conversion_rate"].median())
    remaining_contacts = MAX_TOTAL_CONTACTS
    remaining_budget = float(TOTAL_BUDGET)
    parts: list[pd.DataFrame] = []

    for position, (_, campaign) in enumerate(campaigns.iterrows(), start=1):
        channel = str(campaign["channel"])
        cost_per_contact = float(CHANNELS[channel]["cost_per_contact"])
        segment = apply_filters(profile, campaign).sort_values("ID_NUMBER")
        segment = segment.iloc[:MAX_CUSTOMERS_PER_CAMPAIGN]
        segment = segment.iloc[: max(remaining_contacts, 0)]
        if cost_per_contact > 0:
            affordable = int(remaining_budget // cost_per_contact)
            segment = segment.iloc[: max(affordable, 0)]

        remaining_contacts -= len(segment)
        remaining_budget -= len(segment) * cost_per_contact
        if segment.empty:
            continue

        scored = score_campaign(
            segment,
            str(campaign["target_tariff"]),
            impact_model,
            tariffs,
            fallback_conversion,
            channel,
            _mock_fallback,
        )
        scored = scored.copy()
        scored.insert(0, "campaign_order", position)
        scored.insert(1, "campaign_type", "pilot" if position <= n_pilots else "final")
        scored.insert(2, "campaign_name", str(campaign.get("campaign_name", f"campaign_{position}")))
        scored.insert(3, "target_tariff", str(campaign["target_tariff"]))
        scored.insert(4, "channel", channel)
        scored["contact_cost"] = cost_per_contact
        parts.append(
            scored[
                [
                    "campaign_order",
                    "campaign_type",
                    "campaign_name",
                    "ID_NUMBER",
                    "current_tariff",
                    "arpu_segment",
                    "data_segment",
                    "call_segment",
                    "predicted_arpu",
                    "target_tariff",
                    "channel",
                    "contact_cost",
                    "expected_lift_per_customer",
                ]
            ]
        )

    if not parts:
        return pd.DataFrame()
    ledger = pd.concat(parts, ignore_index=True)
    ledger.insert(0, "contact_number", range(1, len(ledger) + 1))
    return ledger


def _subscriber_summary(profile: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
        "ID_NUMBER",
        "current_tariff",
        "arpu_segment",
        "data_segment",
        "call_segment",
        "predicted_arpu",
    ]
    result = profile[base_columns].copy()
    if ledger.empty:
        result["contact_count"] = 0
        result["campaigns"] = ""
        result["total_contact_cost"] = 0.0
        result["best_expected_lift"] = 0.0
    else:
        grouped = ledger.groupby("ID_NUMBER", sort=False).agg(
            contact_count=("campaign_name", "size"),
            campaigns=("campaign_name", lambda values: "; ".join(dict.fromkeys(values))),
            total_contact_cost=("contact_cost", "sum"),
            best_expected_lift=("expected_lift_per_customer", "max"),
        )
        result = result.merge(grouped, on="ID_NUMBER", how="left")
        result["contact_count"] = result["contact_count"].fillna(0).astype(int)
        result["campaigns"] = result["campaigns"].fillna("")
        result["total_contact_cost"] = result["total_contact_cost"].fillna(0.0)
        result["best_expected_lift"] = result["best_expected_lift"].fillna(0.0)
    result["net_contribution"] = (
        result["best_expected_lift"] - result["total_contact_cost"]
    )
    result["contacted"] = result["contact_count"] > 0
    return result


def generate_visual_report(
    *, seed: int = 42, output_dir: str | Path = ".agent-report"
) -> dict:
    """Run the agent once and write HTML plus detailed CSV artifacts."""

    output = Path(output_dir)
    if not output.is_absolute():
        output = BASE_DIR / output
    output.mkdir(parents=True, exist_ok=True)

    env, internals = make_mock_env(
        seed=seed,
        data_dir=str(BASE_DIR / "data"),
        profile_path=str(BASE_DIR / "customer_profile.csv"),
    )
    final_campaigns = sanitize_campaigns(Agent().act(env), env.tariffs)[:MAX_CAMPAIGNS]
    pilot_campaigns = internals.executed_pilot_campaigns()
    all_rows = pilot_campaigns + final_campaigns
    campaigns = pd.DataFrame(all_rows)
    for column in FILTER_COLUMNS + ["explicit_ids"]:
        if column not in campaigns.columns:
            campaigns[column] = None

    impact_model = _mock_impact_model(pd.read_csv(BASE_DIR / "data" / "change_tariff.csv"))
    profile = env.customer_profile.copy()
    baseline = float(profile["predicted_arpu"].sum())
    result = score_campaigns(
        campaigns,
        profile,
        impact_model,
        env.tariffs,
        baseline,
        _mock_fallback,
        team_id="visual-report",
    )

    ledger = _contact_ledger(
        campaigns,
        profile,
        impact_model,
        env.tariffs,
        len(pilot_campaigns),
    )
    subscribers = _subscriber_summary(profile, ledger)

    campaign_rows = []
    for position, ((_, campaign), detail) in enumerate(
        zip(campaigns.iterrows(), result["campaigns_detail"]), start=1
    ):
        campaign_rows.append(
            {
                "order": position,
                "type": "Пилот" if position <= len(pilot_campaigns) else "Финальная",
                "name": detail["name"],
                "current_tariff": campaign.get("filter_current_tariff") or "все",
                "arpu_segment": campaign.get("filter_arpu_segment") or "все",
                "target_tariff": campaign["target_tariff"],
                "channel": detail["channel"],
                "contacts": detail["n_contacts"],
                "cost": detail["cost"],
                "gross_before_dedup": detail["gross_lift"],
                "net_before_dedup": detail["gross_lift"] - detail["cost"],
                "negative_contacts": detail["n_negative"],
            }
        )
    campaign_summary = pd.DataFrame(campaign_rows)

    pilots = pd.DataFrame(env.pilot_history).copy()
    if not pilots.empty:
        pilots["observed_lift_pct"] = 100.0 * pilots["observed_lift_ratio"]

    segment_summary = (
        profile.groupby("arpu_segment", observed=True)
        .agg(
            subscribers=("ID_NUMBER", "size"),
            average_predicted_arpu=("predicted_arpu", "mean"),
            total_predicted_arpu=("predicted_arpu", "sum"),
        )
        .reset_index()
    )
    segment_summary["audience_share_pct"] = (
        100.0 * segment_summary["subscribers"] / len(profile)
    )

    channel_rows = []
    for name, values in env.channels.items():
        channel_rows.append(
            {
                "Канал": name,
                "Цена контакта": values["cost_per_contact"],
                "Множитель эффективности": values["conversion_multiplier"],
                "Максимум контактов на 100 000": (
                    "без денежного лимита"
                    if values["cost_per_contact"] == 0
                    else int(TOTAL_BUDGET // values["cost_per_contact"])
                ),
            }
        )
    channels = pd.DataFrame(channel_rows)

    ledger_path = output / "contact_ledger.csv"
    subscribers_path = output / "subscriber_summary.csv"
    campaigns_path = output / "campaign_summary.csv"
    report_path = output / "report.html"
    ledger.to_csv(ledger_path, index=False)
    subscribers.to_csv(subscribers_path, index=False)
    campaign_summary.to_csv(campaigns_path, index=False)

    display_campaigns = campaign_summary.rename(
        columns={
            "order": "№",
            "type": "Тип",
            "name": "Кампания",
            "current_tariff": "Текущий тариф",
            "arpu_segment": "ARPU",
            "target_tariff": "Целевой тариф",
            "channel": "Канал",
            "contacts": "Контакты",
            "cost": "Стоимость",
            "gross_before_dedup": "Gross до дедупликации",
            "net_before_dedup": "Net до дедупликации",
            "negative_contacts": "Отрицательных",
        }
    ).copy()
    for column in ["Стоимость", "Gross до дедупликации", "Net до дедупликации"]:
        display_campaigns[column] = display_campaigns[column].map(_format_number)

    display_pilots = pd.DataFrame()
    if not pilots.empty:
        display_pilots = pilots[
            [
                "pilot",
                "target_tariff",
                "channel",
                "n_customers",
                "cost",
                "observed_lift_pct",
                "remaining_budget",
                "remaining_contacts",
            ]
        ].rename(
            columns={
                "pilot": "Пилот",
                "target_tariff": "Целевой тариф",
                "channel": "Канал",
                "n_customers": "Размер",
                "cost": "Стоимость",
                "observed_lift_pct": "Наблюдаемый эффект, %",
                "remaining_budget": "Бюджет после",
                "remaining_contacts": "Контактов осталось",
            }
        )
        display_pilots["Наблюдаемый эффект, %"] = display_pilots[
            "Наблюдаемый эффект, %"
        ].map(lambda value: _format_number(value, 2))
        for column in ["Стоимость", "Бюджет после"]:
            display_pilots[column] = display_pilots[column].map(_format_number)

    contacted_sample = subscribers[subscribers["contacted"]].head(50).copy()
    contacted_sample = contacted_sample[
        [
            "ID_NUMBER",
            "current_tariff",
            "arpu_segment",
            "predicted_arpu",
            "contact_count",
            "campaigns",
            "total_contact_cost",
            "best_expected_lift",
            "net_contribution",
        ]
    ].rename(
        columns={
            "ID_NUMBER": "Абонент",
            "current_tariff": "Текущий тариф",
            "arpu_segment": "ARPU-сегмент",
            "predicted_arpu": "Baseline ARPU",
            "contact_count": "Контактов",
            "campaigns": "Кампании",
            "total_contact_cost": "Стоимость контактов",
            "best_expected_lift": "Лучший gross",
            "net_contribution": "Net-вклад",
        }
    )
    for column in ["Baseline ARPU", "Стоимость контактов", "Лучший gross", "Net-вклад"]:
        contacted_sample[column] = contacted_sample[column].map(_format_number)

    segment_display = segment_summary.rename(
        columns={
            "arpu_segment": "ARPU-сегмент",
            "subscribers": "Абонентов",
            "average_predicted_arpu": "Средний predicted ARPU",
            "total_predicted_arpu": "Суммарный baseline",
            "audience_share_pct": "Доля аудитории, %",
        }
    ).copy()
    segment_display["Средний predicted ARPU"] = segment_display[
        "Средний predicted ARPU"
    ].map(_format_number)
    segment_display["Суммарный baseline"] = segment_display["Суммарный baseline"].map(
        _format_number
    )
    segment_display["Доля аудитории, %"] = segment_display["Доля аудитории, %"].map(
        lambda value: _format_number(value, 1)
    )

    duplicate_contacts = int(result["total_contacts"] - result["unique_customers_targeted"])
    status_class = "good" if result["status"] == "PASS" else "bad"
    html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Byteford United — наглядная проверка</title>
  <style>
    :root {{ --ink:#172126; --muted:#65737b; --line:#dce4e7; --bg:#f4f7f8;
      --card:#fff; --accent:#f3c400; --accent-dark:#c49d00; --good:#16794b; --bad:#b93838; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
      color:var(--ink); background:var(--bg); }}
    main {{ max-width:1440px; margin:auto; padding:32px; }}
    h1 {{ margin:0 0 8px; font-size:34px; }} h2 {{ margin-top:36px; }}
    .subtitle,.muted {{ color:var(--muted); }}
    .banner {{ background:#182328; color:white; padding:28px; border-radius:18px;
      border-left:8px solid var(--accent); }}
    .status {{ display:inline-block; margin-top:14px; padding:6px 12px; border-radius:999px;
      font-weight:700; background:#e7f5ee; color:var(--good); }}
    .status.bad {{ background:#fdeaea; color:var(--bad); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
      gap:14px; margin:18px 0; }}
    .card,.section {{ background:var(--card); border:1px solid var(--line);
      border-radius:14px; padding:18px; box-shadow:0 3px 14px rgba(22,33,38,.04); }}
    .card small {{ display:block; color:var(--muted); }}
    .card strong {{ display:block; margin-top:4px; font-size:25px; }}
    .section {{ margin:18px 0; overflow:auto; }}
    .explain {{ border-left:5px solid var(--accent); }}
    .progress-block {{ margin:16px 0; }}
    .progress-label {{ display:flex; justify-content:space-between; gap:16px; }}
    .progress {{ height:14px; background:#e8edef; border-radius:999px; overflow:hidden; margin-top:7px; }}
    .progress span {{ display:block; height:100%; background:linear-gradient(90deg,var(--accent),var(--accent-dark)); }}
    .progress-note {{ color:var(--muted); font-size:12px; margin-top:4px; }}
    table {{ width:100%; border-collapse:collapse; white-space:nowrap; }}
    th,td {{ text-align:left; padding:9px 11px; border-bottom:1px solid var(--line); }}
    th {{ position:sticky; top:0; background:#edf2f3; font-size:12px; text-transform:uppercase; }}
    tr:hover td {{ background:#fffbea; }}
    code {{ background:#edf2f3; padding:2px 5px; border-radius:4px; }}
    a {{ color:#7a6200; }}
    @media (max-width:700px) {{ main {{ padding:16px; }} h1 {{ font-size:27px; }} }}
  </style>
</head>
<body><main>
  <section class="banner">
    <h1>Наглядная проверка агента</h1>
    <div class="subtitle">Byteford United · mock seed {seed} · локальный отчёт, не судейский балл</div>
    <span class="status {'' if status_class == 'good' else 'bad'}">{escape(result['status'])}</span>
  </section>

  <h2>Что произошло за один запуск</h2>
  <div class="grid">
    <div class="card"><small>Вся аудитория</small><strong>{_format_number(len(profile))}</strong><small>доступных абонентов</small></div>
    <div class="card"><small>Контакты</small><strong>{_format_number(result['total_contacts'])}</strong><small>из {MAX_TOTAL_CONTACTS:,}</small></div>
    <div class="card"><small>Уникальные абоненты</small><strong>{_format_number(result['unique_customers_targeted'])}</strong><small>{result['coverage_pct']:.1f}% аудитории</small></div>
    <div class="card"><small>Повторные контакты</small><strong>{_format_number(duplicate_contacts)}</strong><small>оплачены, но lift не дублируют</small></div>
    <div class="card"><small>Пилоты</small><strong>{len(pilot_campaigns)}</strong><small>из 20 возможных</small></div>
    <div class="card"><small>Финальные кампании</small><strong>{len(final_campaigns)}</strong><small>из 10 возможных</small></div>
  </div>

  <div class="section explain">
    <h3>Что означают 23 441 абонент и 15 000 контактов?</h3>
    <p><strong>{_format_number(len(profile))} абонент</strong> — весь пул, из которого агент может выбирать.
    <strong>{MAX_TOTAL_CONTACTS:,} контактов</strong> — максимальное число доставок предложений, включая пилоты.
    Контакт — это одна строка «одному абоненту показали одну кампанию». Один человек может получить
    несколько контактов: каждый расходует лимит и стоимость, но прирост ARPU засчитывается только один раз —
    по лучшему предложению. Неохваченные абоненты сохраняют baseline и не добавляют ни затрат, ни прироста.</p>
  </div>

  <div class="section">
    <h3>Использование ограничений</h3>
    {_progress('Контакты, включая пилоты', result['total_contacts'], MAX_TOTAL_CONTACTS, 'контактов')}
    {_progress('Денежный бюджет', result['total_cost'], TOTAL_BUDGET, 'у.е.')}
  </div>

  <h2>Экономический результат</h2>
  <div class="grid">
    <div class="card"><small>Baseline без кампаний</small><strong>{_format_number(result['baseline_total_arpu'])}</strong></div>
    <div class="card"><small>Gross lift</small><strong>{_format_number(result['gross_arpu_lift'])}</strong></div>
    <div class="card"><small>Стоимость контактов</small><strong>{_format_number(result['total_cost'])}</strong></div>
    <div class="card"><small>Net lift</small><strong>{_format_number(result['net_arpu_gain'])}</strong><small>{result['growth_vs_baseline_pct']:+.3f}% к baseline</small></div>
    <div class="card"><small>ROI gross / cost</small><strong>{result['roi']:.2f}</strong></div>
    <div class="card"><small>Risk score</small><strong>{result['risk_score_pct']:.1f}%</strong><small>уникальных клиентов с отрицательным эффектом</small></div>
  </div>

  <div class="section explain">
    <h3>Как жюри получает результат?</h3>
    <p>Жюри вызывает <code>Agent.act(env)</code>. Агент тратит часть ресурсов на пилоты и возвращает не более
    10 финальных кампаний. Скорер последовательно применяет фильтры, обрезает превышения лимитов, считает
    персональный процентный lift от <code>predicted_arpu</code>, оставляет каждому абоненту лучший lift,
    а затем вычитает стоимость <em>всех</em> контактов. Цель — максимальный положительный net, а не охват всех
    23 тысяч человек.</p>
  </div>

  <div class="section"><h2>Состав аудитории</h2>{_table(segment_display)}</div>
  <div class="section"><h2>Цена и эффективность каналов</h2>{_table(channels)}</div>
  <div class="section"><h2>Все пилоты и финальные кампании</h2>
    <p class="muted">Gross и net в строках показаны до дедупликации между кампаниями; итог наверху уже дедуплицирован.</p>
    {_table(display_campaigns)}
  </div>
  <div class="section"><h2>Что вернули пилоты</h2>{_table(display_pilots)}</div>
  <div class="section"><h2>Пример первых 50 охваченных абонентов</h2>
    <p class="muted">Полные {len(profile):,} строк находятся в <code>subscriber_summary.csv</code>, а каждая из
    {len(ledger):,} доставок — в <code>contact_ledger.csv</code>.</p>
    {_table(contacted_sample)}
  </div>
  <div class="section">
    <h2>Файлы для ручной проверки</h2>
    <ul>
      <li><a href="contact_ledger.csv">contact_ledger.csv</a> — одна строка на каждый контакт;</li>
      <li><a href="subscriber_summary.csv">subscriber_summary.csv</a> — все абоненты и их итоговый вклад;</li>
      <li><a href="campaign_summary.csv">campaign_summary.csv</a> — пилоты и финальные кампании.</li>
    </ul>
    <p class="muted">Отчёт использует синтетическую mock impact-модель. Он объясняет механику и не раскрывает
    скрытые эффекты судейской среды.</p>
  </div>
</main></body></html>"""
    report_path.write_text(html, encoding="utf-8")

    return {
        "report_path": report_path,
        "ledger_path": ledger_path,
        "subscribers_path": subscribers_path,
        "campaigns_path": campaigns_path,
        "result": result,
        "n_pilots": len(pilot_campaigns),
        "n_final_campaigns": len(final_campaigns),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a visual mock evaluation report")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=".agent-report")
    args = parser.parse_args()
    artifacts = generate_visual_report(seed=args.seed, output_dir=args.output_dir)
    result = artifacts["result"]
    print(f"Статус: {result['status']}")
    print(f"Контактов: {result['total_contacts']:,} из {MAX_TOTAL_CONTACTS:,}")
    print(f"Бюджет: {result['total_cost']:,.0f} из {TOTAL_BUDGET:,}")
    print(f"Net: {result['net_arpu_gain']:,.0f}")
    print(f"HTML-отчёт: {artifacts['report_path']}")
    print(f"Журнал контактов: {artifacts['ledger_path']}")


if __name__ == "__main__":
    main()
