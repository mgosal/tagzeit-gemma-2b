/**
 * Tagzeit CPT Corpus Generator v2
 * Generates synthetic temporal reasoning data with:
 *   - Deterministic Base-60 State Machine [THINK] blocks (verbose + compact modes)
 *   - Shadow Pairs with adjacency guarantee (contrastive learning)
 *   - Human-Fuzzy Time (Temporal Context Anchors with translation step)
 *   - Tokenization Hardening (15% spaced digits, format jitter)
 *   - Hierarchical Temporal Logic (20% multi-unit cascades)
 *   - 12 Human Domains ("Humanity 12")
 *   - MLX-compatible JSONL output
 */

const fs = require('fs');
const { DateTime } = require('luxon');
const { faker } = require('@faker-js/faker');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

const argv = yargs(hideBin(process.argv))
  .option('count',   { type: 'number', default: 5000 })
  .option('output',  { type: 'string', default: 'train.jsonl' })
  .option('eval',    { type: 'string', default: 'eval.jsonl' })
  .option('compact', { type: 'boolean', default: false,
                       describe: 'Use compact THINK blocks (for small models like SmolLM2-135M)' })
  .argv;

const DOMAINS = [
  'Domestic', 'Logistics', 'Professional', 'Wellness',
  'Social', 'Entertainment', 'Parenting', 'Financial',
  'Maintenance', 'History', 'Tech', 'Procrastination'
];

// ---------------------------------------------------------------------------
// Human-Fuzzy Time: Temporal Context Anchors
// ---------------------------------------------------------------------------
const FUZZY_TIMES = [
  { spoken: "half past six",           h: 6,  m: 30 },
  { spoken: "quarter past one",        h: 13, m: 15 },
  { spoken: "quarter to ten",          h: 9,  m: 45 },
  { spoken: "noon",                    h: 12, m: 0  },
  { spoken: "midnight",               h: 0,  m: 0  },
  { spoken: "half past eight",         h: 8,  m: 30 },
  { spoken: "quarter past three",      h: 15, m: 15 },
  { spoken: "ten to five",             h: 16, m: 50 },
  { spoken: "twenty past seven",       h: 7,  m: 20 },
  { spoken: "five to nine",            h: 8,  m: 55 },
  { spoken: "half past eleven",        h: 23, m: 30 },
  { spoken: "quarter to midnight",     h: 23, m: 45 },
];

// ---------------------------------------------------------------------------
// Format Jitter: Hardens tokenization by varying time representation.
// ---------------------------------------------------------------------------
function formatTime(h, m) {
  const hh = String(h).padStart(2, '0');
  const mm = String(m).padStart(2, '0');
  const standard = `${hh}:${mm}`;

  const rand = Math.random();
  if (rand < 0.15) return standard.split('').join(' '); // Spaced: "1 8 : 5 9"
  if (rand < 0.22) return `${hh}.${mm}`;                // Dot: "18.59"
  if (rand < 0.28) {                                     // No-pad: "6:05" or "6:5"
    const nopad_h = String(h);
    const nopad_m = Math.random() < 0.3 ? String(m) : mm;
    return `${nopad_h}:${nopad_m}`;
  }
  if (rand < 0.33) return `${hh}${mm}`;                 // Clump: "1859"
  return standard;
}

function pad2(n) { return String(n).padStart(2, '0'); }

// ---------------------------------------------------------------------------
// VERBOSE [THINK] Block: Deterministic Base-60 State Machine (for Gemma-2-2b)
// 4 steps: Unit Isolation → Overflow Check → Carry Primitive → Zero-Padding
// ---------------------------------------------------------------------------
function generateThinkBlockVerbose(startH, startM, deltaMinutes) {
  const totalM = startM + deltaMinutes;
  const minuteResult = ((totalM % 60) + 60) % 60;
  const hourCarry = Math.floor(totalM / 60);
  const totalH = startH + hourCarry;
  const hourResult = ((totalH % 24) + 24) % 24;

  let t = `[THINK] `;
  t += `M_start = ${startM}. `;
  t += `${startM} ${deltaMinutes >= 0 ? '+' : '-'} ${Math.abs(deltaMinutes)} = ${totalM}. `;
  
  if (totalM >= 60) {
    t += `${totalM} >= 60? Yes. ${totalM} - ${Math.floor(totalM / 60) * 60} = ${pad2(minuteResult)}. Carry ${hourCarry} to Hours. `;
  } else if (totalM < 0) {
    t += `${totalM} < 0? Yes. Borrow from Hours. ${totalM} + ${Math.abs(hourCarry) * 60} = ${pad2(minuteResult)}. Carry ${hourCarry} to Hours. `;
  } else {
    t += `Minutes = ${pad2(minuteResult)}. `;
  }
  
  t += `H_start = ${startH}. ${startH} ${hourCarry >= 0 ? '+' : '-'} ${Math.abs(hourCarry)} = ${totalH}. `;
  
  if (totalH >= 24) {
    t += `${totalH} >= 24? Yes. ${totalH} - 24 = ${pad2(hourResult)}. Day rollover. `;
  } else if (totalH < 0) {
    t += `${totalH} < 0? Yes. ${totalH} + 24 = ${pad2(hourResult)}. Previous day. `;
  } else {
    t += `Hours = ${pad2(hourResult)}. `;
  }
  
  if (minuteResult < 10) t += `Format minute: ${pad2(minuteResult)}. `;
  if (hourResult < 10) t += `Format hour: ${pad2(hourResult)}. `;
  t += `[RESULT] ${pad2(hourResult)}:${pad2(minuteResult)}`;
  return { think: t, resultH: hourResult, resultM: minuteResult };
}

// ---------------------------------------------------------------------------
// COMPACT [THINK] Block: For small models (SmolLM2-135M)
// ~25 tokens instead of ~60. Arithmetic trace is explicit but compressed.
// ---------------------------------------------------------------------------
function generateThinkBlockCompact(startH, startM, deltaMinutes) {
  const totalM = startM + deltaMinutes;
  const minuteResult = ((totalM % 60) + 60) % 60;
  const hourCarry = Math.floor(totalM / 60);
  const totalH = startH + hourCarry;
  const hourResult = ((totalH % 24) + 24) % 24;

  let t = `[THINK] ${startM}${deltaMinutes >= 0 ? '+' : ''}${deltaMinutes}=${totalM}`;
  if (totalM >= 60 || totalM < 0) {
    t += ` mod60=${pad2(minuteResult)} carry=${hourCarry}`;
  }
  t += ` H:${startH}${hourCarry >= 0 ? '+' : ''}${hourCarry}=${totalH}`;
  if (totalH >= 24 || totalH < 0) {
    t += ` mod24=${pad2(hourResult)}`;
  }
  t += ` [RESULT] ${pad2(hourResult)}:${pad2(minuteResult)}`;
  return { think: t, resultH: hourResult, resultM: minuteResult };
}

// ---------------------------------------------------------------------------
// Shadow Pair: base-10 arithmetic record for contrastive learning.
// Uses the SAME numbers as the adjacent temporal record.
// ---------------------------------------------------------------------------
function generateShadowPair(startM, deltaMinutes) {
  const plainSum = startM + deltaMinutes;
  return {
    text: `What is ${startM} + ${deltaMinutes}?\n[THINK] ${startM}+${deltaMinutes}=${plainSum}. [RESULT] ${plainSum}`
  };
}

// ---------------------------------------------------------------------------
// Domain-specific prompt templates
// ---------------------------------------------------------------------------
const DOMAIN_TEMPLATES = {
  Domestic:       (t, d) => d >= 0 ? `I put the roast in the oven at ${t}. It needs ${d} minutes. When is it done?` : `The roast was done at ${t}. It took ${Math.abs(d)} minutes. When did it go in?`,
  Logistics:      (t, d) => d >= 0 ? `The train departs at ${t}. The journey is ${d} minutes. Arrival time?` : `The train arrived at ${t}. The journey was ${Math.abs(d)} minutes. What time did it depart?`,
  Professional:   (t, d) => d >= 0 ? `The meeting starts at ${t} and lasts ${d} minutes. When does it end?` : `The meeting ended at ${t}. It was ${Math.abs(d)} minutes long. When did it start?`,
  Wellness:       (t, d) => d >= 0 ? `Take your medicine at ${t}. The next dose is in ${d} minutes. When?` : `You took your medicine at ${t}. The previous dose was ${Math.abs(d)} minutes ago. What time was that?`,
  Social:         (t, d) => d >= 0 ? `Dinner reservation is at ${t}. If I arrive ${d} minutes late, what time do I get there?` : `I arrived at the restaurant at ${t}, which was ${Math.abs(d)} minutes EARLIER than my reservation. When was the reservation?`,
  Entertainment:  (t, d) => d >= 0 ? `The movie starts at ${t} and is ${d} minutes long. When does it end?` : `The movie ended at ${t}. It was ${Math.abs(d)} minutes long. When did it start?`,
  Parenting:      (t, d) => d >= 0 ? `Baby fell asleep at ${t}. Nap lasts ${d} minutes. When do they wake?` : `Baby woke up at ${t} after a ${Math.abs(d)} minute nap. When did they fall asleep?`,
  Financial:      (t, d) => d >= 0 ? `Market opens at ${t}. I will check prices in ${d} minutes. What time?` : `I checked market prices at ${t}. I was supposed to check them ${Math.abs(d)} minutes ago. What time was that?`,
  Maintenance:    (t, d) => d >= 0 ? `The mechanic started at ${t}. Repair takes ${d} minutes. When is it done?` : `The repair was finished at ${t}. It took ${Math.abs(d)} minutes. When did the mechanic start?`,
  History:        (t, d) => d >= 0 ? `The ceremony began at ${t} and lasted ${d} minutes. When did it end?` : `The ceremony ended at ${t}. It lasted ${Math.abs(d)} minutes. When did it begin?`,
  Tech:           (t, d) => d >= 0 ? `Deploy started at ${t}. Pipeline takes ${d} minutes. When is it live?` : `The site was live at ${t}. Deployment took ${Math.abs(d)} minutes. When did it start?`,
  Procrastination:(t, d) => d >= 0 ? `I said I'd start at ${t} but waited ${d} more minutes. When did I actually start?` : `I actually started at ${t} after waiting ${Math.abs(d)} minutes LESS than I planned. I planned to start at what time?`,
};

// ---------------------------------------------------------------------------
// Record Generator
// ---------------------------------------------------------------------------
function generateRecord(domain, isCascade = false) {
  let startH, startM, deltaMinutes;
  let useFuzzy = false;
  let fuzzyEntry = null;

  if (isCascade) {
    // Multi-Unit Cascade: force boundary conditions
    startH = faker.number.int({ min: 22, max: 23 });
    startM = faker.number.int({ min: 50, max: 59 });
    deltaMinutes = faker.number.int({ min: 1, max: 120 });
  } else if (Math.random() < 0.10) {
    // 10% Human-Fuzzy time
    useFuzzy = true;
    fuzzyEntry = FUZZY_TIMES[faker.number.int({ min: 0, max: FUZZY_TIMES.length - 1 })];
    startH = fuzzyEntry.h;
    startM = fuzzyEntry.m;
    deltaMinutes = faker.number.int({ min: 1, max: 120 });
  } else if (Math.random() < 0.35) {
    // 35% Boundary-focused (minute carry scenarios)
    startH = faker.number.int({ min: 0, max: 23 });
    startM = faker.number.int({ min: 45, max: 59 });
    deltaMinutes = faker.number.int({ min: 1, max: 30 });
  } else {
    // Standard random (including 20% subtraction cases)
    startH = faker.number.int({ min: 0, max: 23 });
    startM = faker.number.int({ min: 0, max: 59 });
    
    if (Math.random() < 0.20) {
        // Subtraction
        deltaMinutes = -faker.number.int({ min: 1, max: 180 });
    } else {
        deltaMinutes = faker.number.int({ min: 1, max: 180 });
    }
  }

  // Generate THINK block (compact or verbose)
  const thinkFn = argv.compact ? generateThinkBlockCompact : generateThinkBlockVerbose;
  const { think, resultH, resultM } = thinkFn(startH, startM, deltaMinutes);

  // Build prompt
  let timeStr;
  let thinkBlock;

  if (useFuzzy) {
    // Temporal Context Anchor: first step translates fuzzy → formal
    timeStr = fuzzyEntry.spoken;
    const formalTime = `${pad2(startH)}:${pad2(startM)}`;
    thinkBlock = think.replace('[THINK] ', `[THINK] '${fuzzyEntry.spoken}' = ${formalTime}. `);
  } else {
    timeStr = formatTime(startH, startM);
    thinkBlock = think;
  }

  const templateFn = DOMAIN_TEMPLATES[domain] || DOMAIN_TEMPLATES.Domestic;
  const prompt = templateFn(timeStr, deltaMinutes);
  const answer = `${pad2(resultH)}:${pad2(resultM)}`;

  // MLX-compatible format: { text: "prompt\nresponse" }
  return {
    text: `${prompt}\n${thinkBlock}\n[ANSWER] ${answer} [/ANSWER]`,
    // Also store structured for potential non-MLX use
    _startM: startM,
    _deltaMinutes: deltaMinutes,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  const trainStream = fs.createWriteStream(argv.output);
  const evalStream = fs.createWriteStream(argv.eval);
  const evalThreshold = 0.05;

  let generated = 0;
  let cascadeCount = 0;
  let shadowCount = 0;
  const cascadeTarget = Math.floor(argv.count * 0.20);

  for (let i = 0; i < argv.count; i++) {
    const domain = DOMAINS[i % DOMAINS.length];

    // Deterministic cascade: every 5th record if under target
    const isCascade = cascadeCount < cascadeTarget && (i % 5 === 0);
    if (isCascade) cascadeCount++;

    // Generate the temporal record first (to get its numbers for shadow pair)
    const record = generateRecord(domain, isCascade);

    const isEval = Math.random() < evalThreshold;
    const stream = isEval ? evalStream : trainStream;

    // Shadow Pair: emit base-10 arithmetic IMMEDIATELY before every 5th temporal record
    // Uses the SAME minute values for contrastive learning within the same attention window
    if (i % 5 === 0) {
      const shadow = generateShadowPair(record._startM, record._deltaMinutes);
      stream.write(JSON.stringify({ text: shadow.text }) + '\n');
      shadowCount++;
    }

    // Emit the temporal record (strip internal metadata)
    stream.write(JSON.stringify({ text: record.text }) + '\n');

    generated++;
    if (generated % 10000 === 0) {
      console.log(`Generated ${generated} records (${cascadeCount} cascades, ${shadowCount} shadows)...`);
    }
  }

  trainStream.end();
  evalStream.end();

  const mode = argv.compact ? 'COMPACT' : 'VERBOSE';
  console.log(`Finished [${mode}]. ${generated} records, ${cascadeCount} cascades (${Math.round(cascadeCount/generated*100)}%), ${shadowCount} shadow pairs.`);
}

main().catch(console.error);
