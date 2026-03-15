/**
 * Tagzeit CPT Corpus Generator v3
 * Generates synthetic temporal reasoning data with:
 *   - Deterministic Linear [THINK] blocks (standardized delimiters)
 *   - Humanized "12 Areas of Usage" (Medicine, TZ offsets, Calendar logic)
 *   - 12h AM/PM support with case/space jitter (1% frequency)
 *   - Semantic inversion fixes across all domains
 *   - EOS tokenization (<|endoftext|>) for training stability
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
                       describe: 'Use compact THINK blocks (for small models)' })
  .argv;

const DOMAINS = [
  'Domestic', 'Logistics', 'Professional', 'Wellness',
  'Social', 'Entertainment', 'Parenting', 'Financial',
  'Maintenance', 'History', 'Tech', 'Procrastination'
];

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

function jitterCase(str) {
  return str.split('').map(char => 
    Math.random() > 0.5 ? char.toUpperCase() : char.toLowerCase()
  ).join('');
}

function formatTime(h, m) {
  const hh = String(h).padStart(2, '0');
  const mm = String(m).padStart(2, '0');
  const standard = `${hh}:${mm}`;
  const rand = Math.random();

  // 35% 12-hour format
  if (rand < 0.35) {
    const period = h >= 12 ? 'PM' : 'AM';
    const h12 = h % 12 || 12;
    const separator = Math.random() > 0.5 ? ' ' : '';
    const dot = Math.random() > 0.7 ? '.' : '';
    
    let ampm = period;
    if (dot) ampm = period.split('').join('.');
    
    // Applying jitter case at only 1% frequency per user request
    if (Math.random() < 0.01) {
      ampm = jitterCase(ampm);
    }
    
    return `${h12}:${mm}${separator}${ampm}`;
  }

  // Jittered 24h formats
  if (rand < 0.50) return standard.split('').join(' '); // Spaced
  if (rand < 0.60) return `${hh}.${mm}`;                // Dot
  if (rand < 0.70) return `${h}:${mm}`;                 // No-pad
  if (rand < 0.75) return `${hh}${mm}`;                 // Clump
  
  return standard;
}

function pad2(n) { return String(n).padStart(2, '0'); }

function build12hTranslation(formatTimeStr, h24, m) {
  const canonical = `${pad2(h24)}:${pad2(m)}`;
  const hasLetters = /[a-z]/i.test(formatTimeStr);
  const isNoon     = /\bnoon\b/i.test(formatTimeStr);
  const isMidnight = /\bmidnight\b/i.test(formatTimeStr);

  // If it's a spoken form or 12h, we need a preamble
  if (hasLetters || isNoon || isMidnight) {
    const label = isNoon ? "Noon" : isMidnight ? "Midnight" : `'${formatTimeStr}'`;
    return { preamble: `${label} = ${canonical}. `, canonicalStr: canonical };
  }

  return { preamble: '', canonicalStr: canonical };
}

function computeTimeAdd(startH, startM, deltaMinutes) {
  const totalM = startM + deltaMinutes;
  const minuteResult = ((totalM % 60) + 60) % 60;
  const hourCarry = Math.floor(totalM / 60);
  const totalH = startH + hourCarry;
  const hourResult = ((totalH % 24) + 24) % 24;
  return { totalM, minuteResult, hourCarry, totalH, hourResult };
}

function generateThinkBlockVerbose(startH, startM, deltaMinutes, opts = {}) {
  const { totalM, minuteResult, hourCarry, totalH, hourResult } = computeTimeAdd(startH, startM, deltaMinutes);
  const displayTime = opts.displayTime || `${pad2(startH)}:${pad2(startM)}`;
  const { preamble } = build12hTranslation(displayTime, startH, startM);

  const absDelta = Math.abs(deltaMinutes);
  const op       = deltaMinutes >= 0 ? '+' : '-';
  let t = `[THINK] ${preamble}M: ${startM} ${op} ${absDelta} = ${totalM}. `;

  if (totalM >= 60) {
    const carriedMinutes = hourCarry * 60;
    t += `${totalM} >= 60 → ${hourCarry} * 60 = ${carriedMinutes}, ${totalM} - ${carriedMinutes} = ${pad2(minuteResult)}. Carry +${hourCarry}h. `;
  } else if (totalM < 0) {
    const borrowHours = Math.abs(hourCarry);
    const borrowedMinutes = borrowHours * 60;
    t += `${totalM} < 0 → borrow ${borrowHours}h. ${borrowHours} * 60 = ${borrowedMinutes}, ${totalM} + ${borrowedMinutes} = ${pad2(minuteResult)}. Carry ${hourCarry}h. `;
  } else {
    t += `No carry. Minutes = ${pad2(minuteResult)}. `;
  }

  const hourOp = hourCarry >= 0 ? '+' : '-';
  t += `H: ${startH} ${hourOp} ${Math.abs(hourCarry)} = ${totalH}. `;

  if (totalH >= 24) {
    t += `${totalH} >= 24 → ${totalH} - 24 = ${pad2(hourResult)}. Next day. `;
  } else if (totalH < 0) {
    t += `${totalH} < 0 → ${totalH} + 24 = ${pad2(hourResult)}. Previous day. `;
  } else {
    t += `Hours = ${pad2(hourResult)}. `;
  }

  if (minuteResult < 10) t += `Pad minute → ${pad2(minuteResult)}. `;
  if (hourResult < 10)   t += `Pad hour → ${pad2(hourResult)}. `;
  
  if (hourResult === 12 && minuteResult === 0) t += `Result is Noon. `;
  if (hourResult === 0 && minuteResult === 0) t += `Result is Midnight. `;

  t += `[/THINK]`;
  return { think: t, resultH: hourResult, resultM: minuteResult };
}

function generateThinkBlockCompact(startH, startM, deltaMinutes, opts = {}) {
  const { totalM, minuteResult, hourCarry, totalH, hourResult } = computeTimeAdd(startH, startM, deltaMinutes);
  const displayTime = opts.displayTime || `${pad2(startH)}:${pad2(startM)}`;
  const { preamble } = build12hTranslation(displayTime, startH, startM);

  let t = `[THINK] ${preamble}${startM}${deltaMinutes >= 0 ? '+' : ''}${deltaMinutes}=${totalM}`;
  if (totalM >= 60 || totalM < 0) {
    const carriedMinutes = Math.abs(hourCarry) * 60;
    t += ` ${Math.abs(hourCarry)}*60=${carriedMinutes} rem=${pad2(minuteResult)} c=${hourCarry}`;
  } else {
    t += ` m=${pad2(minuteResult)}`;
  }

  t += ` | ${startH}${hourCarry >= 0 ? '+' : ''}${hourCarry}=${totalH}`;
  if (totalH >= 24) t += ` -24=${pad2(hourResult)}`;
  else if (totalH < 0) t += ` +24=${pad2(hourResult)}`;
  else t += ` h=${pad2(hourResult)}`;

  t += ` → ${pad2(hourResult)}:${pad2(minuteResult)} [/THINK]`;
  return { think: t, resultH: hourResult, resultM: minuteResult };
}

function generateShadowPair(startM, deltaMinutes) {
  const plainSum = startM + deltaMinutes;
  const op = deltaMinutes >= 0 ? '+' : '-';
  const absDelta = Math.abs(deltaMinutes);
  return {
    text: `What is ${startM} ${op} ${absDelta}?\n[THINK] ${startM}${op}${absDelta}=${plainSum} [/THINK]\n[ANSWER] ${plainSum} [/ANSWER]<|endoftext|>`
  };
}

const DOMAIN_TEMPLATES = {
  Domestic: (t, d) => d >= 0 ? `I put the roast in the oven at ${t}. It needs ${d} minutes. When is it done?` : `The roast was done at ${t}. It cooked for ${Math.abs(d)} minutes. When did it go in?`,
  Logistics: (t, d) => d >= 0 ? `The train departs at ${t}. The journey takes ${d} minutes. What is the arrival time?` : `The train arrived at ${t}. The journey was ${Math.abs(d)} minutes. What time did it depart?`,
  Professional: (t, d) => d >= 0 ? `The meeting starts at ${t} and lasts ${d} minutes. When does it end?` : `The meeting ended at ${t}. It lasted ${Math.abs(d)} minutes. When did it start?`,
  Wellness: (t, d) => {
    const abs = Math.abs(d);
    if (d >= 0) {
      if (abs >= 120 && abs % 60 === 0) return `Take your medication at ${t}. The next dose is every ${abs/60} hours. When is the next dose?`;
      return `Take your medication at ${t}. The next dose is in ${d} minutes. What time is the next dose?`;
    }
    if (abs >= 120 && abs % 60 === 0) return `Your next dose is at ${t}. You take it every ${abs/60} hours. When was the previous dose?`;
    return `You took your medication at ${t}. The previous dose was ${abs} minutes before that. What time was the previous dose?`;
  },
  Social: (t, d) => {
    // Arrival and Reservation logic (Fixed inversion)
    if (d >= 0) return `My dinner reservation is at ${t}. I arrived ${d} minutes late. What time did I arrive?`;
    return `My dinner reservation is at ${t}. I arrived ${Math.abs(d)} minutes early. What time did I arrive?`;
  },
  Entertainment: (t, d) => d >= 0 ? `The movie starts at ${t} and runs for ${d} minutes. When does it end?` : `The movie ended at ${t}. It was ${Math.abs(d)} minutes long. When did it start?`,
  Parenting: (t, d) => d >= 0 ? `Baby fell asleep at ${t}. The nap lasts ${d} minutes. When do they wake up?` : `Baby woke up at ${t} after a ${Math.abs(d)} minute nap. When did they fall asleep?`,
  Financial: (t, d) => d >= 0 ? `The market opens at ${t}. I plan to check prices ${d} minutes after open. What time is that?` : `I checked the market at ${t}. The market opened ${Math.abs(d)} minutes before that. When did it open?`,
  Maintenance: (t, d) => d >= 0 ? `The mechanic started work at ${t}. The repair takes ${d} minutes. When is it done?` : `The repair was finished at ${t}. It took ${Math.abs(d)} minutes. When did the mechanic start?`,
  History: (t, d) => {
    const abs = Math.abs(d);
    if (d >= 0) {
      if (abs > 720) return `The march began at ${t} and lasted ${d} minutes (crossing into the next day). When did it end?`;
      return `The ceremony began at ${t} and lasted ${d} minutes. When did it end?`;
    }
    if (abs > 720) return `The siege ended at ${t}. It lasted ${abs} minutes, starting the previous day. When did it begin?`;
    return `The ceremony ended at ${t}. It lasted ${abs} minutes. When did it begin?`;
  },
  Tech: (t, d) => {
    const abs = Math.abs(d);
    if (d >= 0) {
      if (abs % 60 === 0 && abs <= 720 && Math.random() < 0.3) {
        return `The server timestamp shows ${t} UTC. The client is UTC+${abs/60}. What time does the client see?`;
      }
      return `Deployment started at ${t}. The pipeline takes ${d} minutes. When does it go live?`;
    }
    if (abs % 60 === 0 && abs <= 720 && Math.random() < 0.3) {
      return `The client shows ${t} local time (UTC+${abs/60}). What is the current UTC time?`;
    }
    return `The site went live at ${t}. Deployment took ${abs} minutes. When did deployment start?`;
  },
  Procrastination: (t, d) => {
    if (d >= 0) return `I planned to start working at ${t} but procrastinated for ${d} more minutes. When did I actually start?`;
    return `I actually started working at ${t}, which was ${Math.abs(d)} minutes after my planned start. When did I plan to start?`;
  },
};

function generateRecord(domain, isCascade = false, hourOverride = null) {
  let startH = hourOverride !== null ? hourOverride : faker.number.int({ min: 0, max: 23 });
  let startM = faker.number.int({ min: 0, max: 50 });
  let deltaMinutes;
  
  const roll = Math.random();
  let useFuzzy = false;
  let fuzzyEntry = null;

  if (isCascade) {
    startH = Math.max(startH, 22);
    startM = faker.number.int({ min: 50, max: 59 });
    deltaMinutes = faker.number.int({ min: 1, max: 120 });
  } else if (roll < 0.10) {
    useFuzzy = true;
    fuzzyEntry = FUZZY_TIMES[faker.number.int({ min: 0, max: FUZZY_TIMES.length - 1 })];
    startH = fuzzyEntry.h;
    startM = fuzzyEntry.m;
    deltaMinutes = faker.number.int({ min: 1, max: 120 });
  } else if (roll < 0.45) {
    startM = faker.number.int({ min: 45, max: 59 });
    deltaMinutes = faker.number.int({ min: 1, max: 30 });
  } else {
    deltaMinutes = (Math.random() < 0.2 ? -1 : 1) * faker.number.int({ min: 1, max: 180 });
  }

  const thinkFn = argv.compact ? generateThinkBlockCompact : generateThinkBlockVerbose;
  const timeStr = useFuzzy ? fuzzyEntry.spoken : formatTime(startH, startM);
  const { think, resultH, resultM } = thinkFn(startH, startM, deltaMinutes, { displayTime: timeStr });

  const templateFn = DOMAIN_TEMPLATES[domain] || DOMAIN_TEMPLATES.Domestic;
  const prompt = templateFn(timeStr, deltaMinutes);
  const answer = (resultH === 12 && resultM === 0) ? "noon" : (resultH === 0 && resultM === 0) ? "midnight" : `${pad2(resultH)}:${pad2(resultM)}`;

  return {
    text: `${prompt}\n${think}\n[ANSWER] ${answer} [/ANSWER]<|endoftext|>`,
    _startM: startM,
    _deltaMinutes: deltaMinutes
  };
}

async function main() {
  const trainStream = fs.createWriteStream(argv.output);
  const evalStream = fs.createWriteStream(argv.eval);
  const evalThreshold = 0.05;

  let generated = 0;
  let hourPointer = 0;

  for (let i = 0; i < argv.count; i++) {
    const domain = DOMAINS[i % DOMAINS.length];
    const hourOverride = hourPointer % 24;
    hourPointer++;
    
    const isCascade = (i % 5 === 0);
    const record = generateRecord(domain, isCascade, hourOverride);

    const stream = Math.random() < evalThreshold ? evalStream : trainStream;

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
