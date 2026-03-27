/**
 * Domain-Typed Tokenizer — Temporal Corpus Generator
 * ====================================================
 * Generates synthetic training data for the Route-to-Luxon architecture.
 *
 * Output format teaches the LLM to:
 *   1. Understand natural language temporal queries
 *   2. Emit structured [ROUTE_*] calls instead of computing answers
 *
 * Features (inherited from Tagzeit experiment + new):
 *   - 12 everyday semantic domains
 *   - Format Jitter (~11%), Shadow Pairs, Subtraction (~20%)
 *   - Fuzzy/spoken time support
 *   - [NEW] Routing output format ([ROUTE_TIME_ADD], [ROUTE_TIME_SUB])
 *   - [NEW] Negative examples (non-temporal numbers)
 *   - [NEW] Typed token format ([HEAD_TIME], [ARG_HOUR_*], [ARG_MIN_*])
 */

const fs = require('fs');
const { DateTime } = require('luxon');
const { faker } = require('@faker-js/faker');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

const argv = yargs(hideBin(process.argv))
  .option('count',   { type: 'number', default: 5000 })
  .option('output',  { type: 'string', default: 'train_routed.jsonl' })
  .option('eval',    { type: 'string', default: 'eval_routed.jsonl' })
  .option('negatives', { type: 'number', default: 0.05,
                         describe: 'Fraction of negative (non-temporal) examples' })
  .argv;

const DOMAINS = [
  'Domestic', 'Logistics', 'Professional', 'Wellness',
  'Social', 'Entertainment', 'Parenting', 'Financial',
  'Maintenance', 'History', 'Tech', 'Procrastination'
];

const FUZZY_TIMES = [
  { spoken: "half past six",       h: 18, m: 30 },
  { spoken: "quarter past one",    h: 13, m: 15 },
  { spoken: "quarter to ten",      h: 21, m: 45 },
  { spoken: "noon",                h: 12, m: 0  },
  { spoken: "midnight",            h: 0,  m: 0  },
  { spoken: "half past eight",     h: 20, m: 30 },
  { spoken: "quarter past three",  h: 15, m: 15 },
  { spoken: "ten to five",         h: 16, m: 50 },
  { spoken: "twenty past seven",   h: 19, m: 20 },
  { spoken: "five to nine",        h: 20, m: 55 },
  { spoken: "half past eleven",    h: 23, m: 30 },
  { spoken: "quarter to midnight", h: 23, m: 45 },
  { spoken: "twenty past two",     h: 14, m: 20 },
  { spoken: "five past three",     h: 15, m: 5  },
  { spoken: "ten past four",       h: 16, m: 10 },
];

const NEGATIVE_TEMPLATES = [
  (n) => ({ q: `I live in room ${n} at the hotel. How many rooms are on the floor?`, a: `[NO_ROUTE] This is not a temporal question.` }),
  (n) => ({ q: `The flight number is BA${n}. Is this a morning flight?`, a: `[NO_ROUTE] Flight numbers are identifiers, not times.` }),
  (n) => ({ q: `Highway ${n} is closed. Which route should I take?`, a: `[NO_ROUTE] This is a navigation question, not temporal.` }),
  (n) => ({ q: `There were ${n} people at the event. How many tables?`, a: `[NO_ROUTE] This is a quantity question, not temporal.` }),
  (n) => ({ q: `The temperature was ${n} degrees. Will it rain?`, a: `[NO_ROUTE] This is a weather question, not temporal.` }),
  (n) => ({ q: `It cost £${n} for the ticket. How much for two?`, a: `[NO_ROUTE] This is a financial question, not temporal.` }),
  (n) => ({ q: `The package weighs ${n} grams. Too heavy to post?`, a: `[NO_ROUTE] This is a weight question, not temporal.` }),
  (n) => ({ q: `Turn to page ${n} in the manual.`, a: `[NO_ROUTE] This is a reference, not temporal.` }),
];

function pad2(n) { return String(n).padStart(2, '0'); }

function timeTokens(h, m) {
  return `[HEAD_TIME] [ARG_HOUR_${pad2(h)}] [ARG_MIN_${pad2(m)}]`;
}

function durationTokens(deltaMinutes) {
  const absDelta = Math.abs(deltaMinutes);
  const hours = Math.floor(absDelta / 60);
  const mins = absDelta % 60;
  let tokens = '[HEAD_DURATION]';
  if (hours > 0) tokens += ` [ARG_HOUR_${pad2(hours)}]`;
  if (mins > 0 || hours === 0) tokens += ` [ARG_MIN_${pad2(mins)}]`;
  return tokens;
}

function formatTimeHuman(h, m) {
  const hh = pad2(h);
  const mm = pad2(m);
  const standard = `${hh}:${mm}`;
  const rand = Math.random();
  if (rand < 0.35) {
    const period = h >= 12 ? 'PM' : 'AM';
    const h12 = h % 12 || 12;
    const sep = Math.random() > 0.5 ? ' ' : '';
    return `${h12}:${mm}${sep}${period}`;
  }
  if (rand < 0.46) return standard.split('').join(' ');
  if (rand < 0.56) return `${hh}.${mm}`;
  if (rand < 0.66) return `${h}:${mm}`;
  return standard;
}

const DOMAIN_TEMPLATES = {
  Domestic:        (t, d) => d >= 0 ? `I put the roast in at ${t}. It needs ${Math.abs(d)} minutes. When is it done?` : `The roast was done at ${t}. It cooked for ${Math.abs(d)} minutes. When did it go in?`,
  Logistics:       (t, d) => d >= 0 ? `The train departs at ${t}. Journey takes ${Math.abs(d)} minutes. Arrival time?` : `Train arrived at ${t}. Journey was ${Math.abs(d)} minutes. Departure time?`,
  Professional:    (t, d) => d >= 0 ? `Meeting starts at ${t}, lasts ${Math.abs(d)} minutes. When does it end?` : `Meeting ended at ${t}. It lasted ${Math.abs(d)} minutes. When did it start?`,
  Wellness:        (t, d) => d >= 0 ? `Take medication at ${t}. Next dose in ${Math.abs(d)} minutes. What time?` : `Took medication at ${t}. Previous dose was ${Math.abs(d)} minutes before. What time?`,
  Social:          (t, d) => d >= 0 ? `Reservation at ${t}. Arrived ${Math.abs(d)} minutes late. What time?` : `Reservation at ${t}. Arrived ${Math.abs(d)} minutes early. What time?`,
  Entertainment:   (t, d) => d >= 0 ? `Movie starts at ${t}, runs ${Math.abs(d)} minutes. When does it end?` : `Movie ended at ${t}. It was ${Math.abs(d)} minutes. When did it start?`,
  Parenting:       (t, d) => d >= 0 ? `Baby fell asleep at ${t}. Nap lasts ${Math.abs(d)} minutes. Wake time?` : `Baby woke at ${t} after ${Math.abs(d)} minute nap. When did they sleep?`,
  Financial:       (t, d) => d >= 0 ? `Market opens at ${t}. Check prices ${Math.abs(d)} minutes later. What time?` : `Checked market at ${t}. It opened ${Math.abs(d)} minutes before. When?`,
  Maintenance:     (t, d) => d >= 0 ? `Mechanic started at ${t}. Repair takes ${Math.abs(d)} minutes. Done when?` : `Repair done at ${t}. Took ${Math.abs(d)} minutes. When did it start?`,
  History:         (t, d) => d >= 0 ? `Ceremony began at ${t}, lasted ${Math.abs(d)} minutes. When did it end?` : `Ceremony ended at ${t}. Lasted ${Math.abs(d)} minutes. When did it begin?`,
  Tech:            (t, d) => d >= 0 ? `Deploy started at ${t}. Pipeline takes ${Math.abs(d)} minutes. Live when?` : `Site live at ${t}. Deploy took ${Math.abs(d)} minutes. When did it start?`,
  Procrastination: (t, d) => d >= 0 ? `Planned to start at ${t}, procrastinated ${Math.abs(d)} minutes. Actual start?` : `Started at ${t}, ${Math.abs(d)} minutes after planned. Planned start?`,
};

function computeResult(startH, startM, deltaMinutes) {
  const dt = DateTime.fromObject({ hour: startH, minute: startM });
  const result = deltaMinutes >= 0
    ? dt.plus({ minutes: Math.abs(deltaMinutes) })
    : dt.minus({ minutes: Math.abs(deltaMinutes) });
  return { hour: result.hour, minute: result.minute };
}

function generateRoutedRecord(domain) {
  let startH = faker.number.int({ min: 0, max: 23 });
  let startM = faker.number.int({ min: 0, max: 59 });
  let deltaMinutes;
  let useFuzzy = false;
  let fuzzyEntry = null;
  const roll = Math.random();

  if (roll < 0.10) {
    useFuzzy = true;
    fuzzyEntry = FUZZY_TIMES[faker.number.int({ min: 0, max: FUZZY_TIMES.length - 1 })];
    startH = fuzzyEntry.h;
    startM = fuzzyEntry.m;
    deltaMinutes = faker.number.int({ min: 1, max: 120 });
  } else if (roll < 0.45) {
    startM = faker.number.int({ min: 45, max: 59 });
    deltaMinutes = faker.number.int({ min: 1, max: 30 });
  } else if (roll < 0.65) {
    deltaMinutes = -1 * faker.number.int({ min: 1, max: 180 });
  } else {
    deltaMinutes = faker.number.int({ min: 1, max: 180 });
  }

  const timeStr = useFuzzy ? fuzzyEntry.spoken : formatTimeHuman(startH, startM);
  const templateFn = DOMAIN_TEMPLATES[domain] || DOMAIN_TEMPLATES.Domestic;
  const prompt = templateFn(timeStr, deltaMinutes);
  const result = computeResult(startH, startM, deltaMinutes);
  const routeOp = deltaMinutes >= 0 ? '[ROUTE_TIME_ADD]' : '[ROUTE_TIME_SUB]';
  const routingCall = `${routeOp} ${timeTokens(startH, startM)} ${durationTokens(deltaMinutes)}`;
  const resultStr = `${pad2(result.hour)}:${pad2(result.minute)}`;

  return {
    text: `${prompt}\n[ROUTE] ${routingCall} [/ROUTE]`,
    _startM: startM,
    _deltaMinutes: deltaMinutes,
  };
}

function generateShadowPair(startM, deltaMinutes) {
  const plainSum = startM + deltaMinutes;
  const op = deltaMinutes >= 0 ? '+' : '-';
  const absDelta = Math.abs(deltaMinutes);
  return {
    text: `What is ${startM} ${op} ${absDelta}?\n[NO_ROUTE] This is base-10 arithmetic, not temporal.`
  };
}

function generateNegativeExample() {
  const templateFn = NEGATIVE_TEMPLATES[faker.number.int({ min: 0, max: NEGATIVE_TEMPLATES.length - 1 })];
  const n = faker.number.int({ min: 100, max: 9999 });
  const { q, a } = templateFn(n);
  return { text: `${q}\n${a}` };
}

async function main() {
  const trainStream = fs.createWriteStream(argv.output);
  const evalStream = fs.createWriteStream(argv.eval);
  const evalThreshold = 0.05;

  let generated = 0;

  for (let i = 0; i < argv.count; i++) {
    const domain = DOMAINS[i % DOMAINS.length];
    const stream = Math.random() < evalThreshold ? evalStream : trainStream;

    // Negative example injection
    if (Math.random() < argv.negatives) {
      const neg = generateNegativeExample();
      stream.write(JSON.stringify({ text: neg.text }) + '\n');
      generated++;
      continue;
    }

    const record = generateRoutedRecord(domain);

    // Shadow pair (every 5th record)
    if (i % 5 === 0) {
      const shadow = generateShadowPair(record._startM, record._deltaMinutes);
      stream.write(JSON.stringify({ text: shadow.text }) + '\n');
    }

    stream.write(JSON.stringify({ text: record.text }) + '\n');
    generated++;

    if (generated % 1000 === 0) console.log(`Generated ${generated} records...`);
  }

  trainStream.end();
  evalStream.end();
  console.log(`Finished. ${generated} records generated.`);
}

main().catch(console.error);
