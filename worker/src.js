// AI_FX の時計役 (Cloudflare Workers の定期実行)。
// GitHub の定期実行は混雑で大きく間引かれるため、Cloudflare から GitHub の実行ボタン (workflow_dispatch) を押す。
// 押すだけで、分析そのものは GitHub Actions で動く。
//   5分ごと : 5分監視 (fx-quick.yml)          … 為替市場が開いている時間だけ
//   30分ごと: 毎時の分析・朝の予想 (fx-auto.yml) … 同上。土曜 10:15〜19 (日本時間) は週次の見直しのため1回押す
//   1時間ごと: 週末の速報の見張り (news-watch.yml) … 市場が閉まっている間だけ (毎時 37〜41分)
// 必要な秘密の値 (Cloudflare の Worker の設定 → 変数とシークレット): GH_TOKEN (GitHub のアクセストークン)

const REPO = "ke104-debug/fx-";

function marketOpen(d) {
  // 為替市場: 日曜 21:00 UTC 〜 金曜 22:00 UTC (夏冬で1時間ずれるが、広めに取る)
  const wd = d.getUTCDay(), h = d.getUTCHours();
  if (wd === 6) return false;
  if (wd === 0 && h < 20) return false;
  if (wd === 5 && h >= 22) return false;
  return true;
}

async function dispatch(env, workflow) {
  const r = await fetch(`https://api.github.com/repos/${REPO}/actions/workflows/${workflow}/dispatches`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.GH_TOKEN}`,
      "Accept": "application/vnd.github+json",
      "User-Agent": "ai-fx-clock",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: "main" }),
  });
  return `${workflow}: ${r.status}`;
}

async function tick(env, when) {
  const d = new Date(when), m = d.getUTCMinutes(), out = [];
  const weekly = d.getUTCDay() === 6 && d.getUTCHours() === 1 && m >= 15 && m < 20;   // 土曜 10:15 日本時間
  if (marketOpen(d)) {
    out.push(await dispatch(env, "fx-quick.yml"));
    if ((m >= 15 && m < 20) || (m >= 45 && m < 50)) out.push(await dispatch(env, "fx-auto.yml"));
  } else {
    if (weekly) out.push(await dispatch(env, "fx-auto.yml"));
    if (m >= 37 && m < 42) out.push(await dispatch(env, "news-watch.yml"));   // 土日もニュースを見張る (利用者の指示 2026-11-28)
  }
  return out;
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(tick(env, event.scheduledTime).then((r) => console.log(r.join(" / ") || "市場が閉まっているため何もしない")));
  },
  // ブラウザで開くと状態を確認できる (秘密の値は表示しない)
  async fetch(request, env) {
    const now = new Date();
    return new Response(JSON.stringify({
      name: "AI_FX の時計役", time_utc: now.toISOString(), market_open: marketOpen(now),
      token_set: Boolean(env.GH_TOKEN), repo: REPO,
    }, null, 2), { headers: { "content-type": "application/json; charset=utf-8" } });
  },
};
