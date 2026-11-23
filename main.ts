// Trend + pullback bot for the cbBTC/USDC 24h race (make.fun btc-race-24h-20261123-0511).
//
// How it works, in plain terms: the harness sends one `hello` line, then one
// `batch` per minute `bar` (the bar that just closed, the fills of the previous
// batch's `orders`, and the account). The bot answers every batch with exactly
// one reply. An order decided on a batch fills at the next bar's open, so every
// decision is made on the latest close and the fills land one bar later.
//
// Strategy: ride the trend, buy the dip inside it, cut losses fast.
//   * Trend filter: fast EMA (60 bars) above slow EMA (240 bars) — slow
//     enough that one-minute noise cannot flip it.
//   * Trend entry: price above the fast EMA, 60-bar return above +0.2%, RSI
//     not hot, and the slow EMA itself rising over the last 120 bars.
//   * Pullback entry: same uptrend, but price at least 0.8% below its 60-bar
//     high and RSI washed out (< 35) — a mean-reversion buy inside the trend.
//   * Exits: a 2% trailing stop off the highest close since entry (arms only
//     after +1%), or a 1.5% hard stop below the entry close. No exit on an
//     EMA flip: on one-minute bars the fast EMA crosses the slow one on mere
//     chop, and selling there locks in noise plus ~26 bps of costs each time.
//     The stops alone cap the loss on any single trade near 2%.
//   * One position only, full size on entry (long only, no leverage), a 60-bar
//     cooldown after every exit so a whipsaw cannot re-buy instantly.
//   * A full round trip costs ~26 bps in fees, slippage and impact, so the bot
//     only trades multi-bar moves: it holds for hours, not minutes.
// Deterministic: no wall clock, no Math.random, no environment reads. All
// randomness-free; the only seed the bot ever sees is hello.seed, unused.

import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

const SYMBOL = 'CBBTC';

// --- tunables (all in basis points unless noted) ---------------------------
const FAST_LEN = 60; // fast EMA length, in bars (1h) — slow enough to ignore 1m noise
const SLOW_LEN = 240; // slow EMA length, in bars (4h)
const RSI_LEN = 14; // Wilder RSI length
const LOOKBACK = 60; // bars for the momentum / pullback window
const MIN_HISTORY = 120; // bars of closes before the first trade (2h: the slow EMA is still warming up, so the rising-slow-EMA check guards entries)
const MAX_SLIPPAGE_BPS = 200; // vs the latest close; a normal 1m move is ~5 bps
const TRAIL_BPS = 200; // trailing stop: 2.0% below the peak close since entry
const TRAIL_ARM_BPS = 100; // trailing stop only arms once price rose 1% above entry
const HARD_STOP_BPS = 150; // hard stop: 1.5% below the entry close
const PULLBACK_BPS = 80; // pullback entry needs >= 0.8% off the 60-bar high
const RSI_HOT = 75; // never chase a trend entry above this RSI
const RSI_WASHED = 35; // pullback entry needs RSI at or below this
const TREND_MIN_BPS = 20; // trend entry needs >= 0.2% gain over the last 60 bars
const SLOW_RISE_BARS = 120; // entry also needs the slow EMA above its value this many bars ago
const COOLDOWN_BARS = 60; // bars to wait after an exit before re-entering

// --- small indicator helpers (plain numbers; prices are ~1e5, floats are fine
// for signals — every money decision still goes through the shim's bigints) ---

/** Exponential moving average step: alpha = 2 / (len + 1). */
function emaStep(prev: number, price: number, len: number): number {
  const alpha = 2 / (len + 1);
  return price * alpha + prev * (1 - alpha);
}

/** Wilder RSI step. Returns the new RSI and the updated average gain/loss. */
function rsiStep(
  prevGain: number,
  prevLoss: number,
  change: number,
  len: number,
): { rsi: number; gain: number; loss: number } {
  const gain = Math.max(change, 0);
  const loss = Math.max(-change, 0);
  const avgGain = (prevGain * (len - 1) + gain) / len;
  const avgLoss = (prevLoss * (len - 1) + loss) / len;
  const rsi = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  return { rsi, gain: avgGain, loss: avgLoss };
}

// --- bot state ---------------------------------------------------------------

interface BotState {
  closes: number[]; // close of every CBBTC bar seen, in USDC
  slowHist: number[]; // slow EMA value after each bar, for the rising-trend check
  emaFast: number; // EMA(FAST_LEN) of closes
  emaSlow: number; // EMA(SLOW_LEN) of closes
  rsi: number; // Wilder RSI(RSI_LEN)
  avgGain: number; // RSI smoothing state
  avgLoss: number; // RSI smoothing state
  holding: boolean; // true once a buy fill has been seen in the account
  entryClose: number; // close of the bar on which the current entry was decided
  peakClose: number; // highest close seen while holding
  lastExitSeq: number; // seq of the batch on which we last sold (cooldown)
  trades: number; // filled buys counted from fills (diagnostics only)
}

function freshState(): BotState {
  return {
    closes: [],
    slowHist: [],
    emaFast: NaN,
    emaSlow: NaN,
    rsi: 50,
    avgGain: 0,
    avgLoss: 0,
    holding: false,
    entryClose: 0,
    peakClose: 0,
    lastExitSeq: -1_000_000,
    trades: 0,
  };
}

/** Fold one bar's close into the indicators. Call once per CBBTC bar. */
function observeClose(s: BotState, close: number): void {
  const prev = s.closes.length > 0 ? s.closes[s.closes.length - 1] : close;
  s.closes.push(close);
  if (Number.isNaN(s.emaFast)) {
    s.emaFast = close;
    s.emaSlow = close;
    return;
  }
  s.emaFast = emaStep(s.emaFast, close, FAST_LEN);
  s.emaSlow = emaStep(s.emaSlow, close, SLOW_LEN);
  s.slowHist.push(s.emaSlow);
  if (s.closes.length > 1) {
    const stepped = rsiStep(s.avgGain, s.avgLoss, close - prev, RSI_LEN);
    s.rsi = stepped.rsi;
    s.avgGain = stepped.gain;
    s.avgLoss = stepped.loss;
  }
}

/** Highest close over the last `n` bars (fewer while warming up). */
function rollingHigh(s: BotState, n: number): number {
  const start = Math.max(0, s.closes.length - n);
  let high = -Infinity;
  for (let i = start; i < s.closes.length; i++) {
    if (s.closes[i] > high) high = s.closes[i];
  }
  return high;
}

/** Return over the last `n` bars, in basis points. */
function nBarReturnBps(s: BotState, n: number): number {
  if (s.closes.length <= n) return 0;
  const now = s.closes[s.closes.length - 1];
  const then = s.closes[s.closes.length - 1 - n];
  return ((now - then) / then) * 10_000;
}

// --- decision logic ----------------------------------------------------------

/**
 * Decide the orders for one batch. `state` is the shim's live view (cash,
 * positions, latest bar) and `batch` is the bar that just closed. Returns at
 * most one order: a full-size buy when flat and a trigger fires, or a full
 * sell when holding and an exit fires.
 */
function decide(state: BarsState, batch: Batch, s: BotState): OrderInput[] {
  // Warm-up batches carry bars from before the scored window and any order
  // decided on one is rejected, so never trade on them.
  if (state.warmingUp) return [];

  // Fold this batch's CBBTC bar into the indicators (at most one per batch).
  for (const bar of batch.bars) {
    if (bar.symbol !== SYMBOL) continue;
    const scale = state.symbols.get(SYMBOL)?.price_scale ?? 5;
    observeClose(s, Number(BigInt(bar.c)) / 10 ** scale);
  }

  // Count filled buys for the diagnostics line (fills arrive a batch late).
  for (const fill of batch.fills) {
    if (fill.status === 'filled' && fill.side === 'buy') s.trades += 1;
  }

  // Sync the holding flag with the account the harness reports: a position
  // appears the batch after a buy fills, and vanishes after a sell fills.
  const pos = state.positions.get(SYMBOL);
  const qty = pos?.qty ?? 0n;
  const wasHolding = s.holding;
  s.holding = qty > 0n;
  if (s.holding && !wasHolding) {
    // A fresh fill: anchor the peak at the entry close for the trailing stop.
    s.peakClose = s.entryClose > 0 ? s.entryClose : s.closes[s.closes.length - 1];
  }
  if (!s.holding) {
    s.entryClose = 0;
    s.peakClose = 0;
  }

  if (s.closes.length < MIN_HISTORY) return [];
  const close = s.closes[s.closes.length - 1];
  if (s.holding && close > s.peakClose) s.peakClose = close;

  const uptrend = s.emaFast > s.emaSlow;

  // --- exit: sell everything when holding ----------------------------------
  if (s.holding) {
    // The trailing stop only arms once the trade has run 1% in our favour,
    // so ordinary chop right after entry cannot stop us out.
    const trailArmed = s.entryClose > 0 && s.peakClose >= s.entryClose * (1 + TRAIL_ARM_BPS / 10_000);
    const trailHit = trailArmed && close <= s.peakClose * (1 - TRAIL_BPS / 10_000);
    const hardStop = s.entryClose > 0 && close <= s.entryClose * (1 - HARD_STOP_BPS / 10_000);
    if (trailHit || hardStop) {
      s.lastExitSeq = state.seq;
      return [{ side: 'sell', symbol: SYMBOL, qty, max_slippage_bps: MAX_SLIPPAGE_BPS }];
    }
    return [];
  }

  // --- entry: flat, in an uptrend ------------------------------------------
  if (state.seq - s.lastExitSeq < COOLDOWN_BARS) return [];
  if (!uptrend) return [];
  // The long trend itself must be rising, not just flat above the price:
  // the slow EMA has to sit above where it was SLOW_RISE_BARS ago.
  const slowRising =
    s.slowHist.length > SLOW_RISE_BARS &&
    s.emaSlow > s.slowHist[s.slowHist.length - 1 - SLOW_RISE_BARS];
  if (!slowRising) return [];

  const momentumBps = nBarReturnBps(s, LOOKBACK);
  const highLookback = rollingHigh(s, LOOKBACK);
  const pullbackBps = highLookback > 0 ? ((highLookback - close) / highLookback) * 10_000 : 0;

  const trendEntry =
    close > s.emaFast && momentumBps > TREND_MIN_BPS && s.rsi < RSI_HOT;
  const pullbackEntry =
    momentumBps > 0 && pullbackBps >= PULLBACK_BPS && s.rsi <= RSI_WASHED;

  if (!trendEntry && !pullbackEntry) return [];

  // Full size: spend all cash (the weight cap is 100% of equity, and when
  // flat cash == equity, so this is the whole account on one trade).
  const notional = state.cash;
  if (notional <= 0n) return [];
  s.entryClose = close;
  return [{ side: 'buy', symbol: SYMBOL, notional_quote_units: notional, max_slippage_bps: MAX_SLIPPAGE_BPS }];
}

// --- wire it up ---------------------------------------------------------------

const botState = freshState();

await runBars({
  onBatch(state, batch) {
    return decide(state, batch, botState);
  },
  onDone(state) {
    const equity = Number(state.equity) / 1_000_000;
    console.error(
      `done: equity=${equity.toFixed(2)} USDC trades=${botState.trades} bars=${botState.closes.length}`,
    );
  },
});
