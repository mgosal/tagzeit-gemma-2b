#!/usr/bin/env node
/**
 * Tagzeit Route-Mode Training Data Generator v1
 * ================================================
 * Generates synthetic temporal reasoning data in [ROUTE] token format
 * for the Route-to-Luxon SFT pipeline.
 *
 * Output format:
 *   Q: <natural language question>
 *   A: [ROUTE] [ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_HH] [ARG_MIN_MM] [HEAD_DURATION] [ARG_HOUR_HH] [ARG_MIN_MM] [/ROUTE]
 *
 * Features:
 *   - 19 domains, 120+ templates
 *   - Balanced ADD/SUB/DURATION_BETWEEN/NO_ROUTE
 *   - Fuzzy/colloquial time expressions
 *   - Varied duration phrasing
 *   - 12h/24h format jitter
 *
 * Usage:
 *   node generator_route.js --count 100000 --output train_routed.jsonl --eval eval_routed.jsonl
 */

const fs = require('fs');
const { DateTime } = require('luxon');
const { faker } = require('@faker-js/faker');
const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');

const argv = yargs(hideBin(process.argv))
  .option('count',  { type: 'number', default: 100000 })
  .option('output', { type: 'string', default: 'train_routed.jsonl' })
  .option('eval',   { type: 'string', default: 'eval_routed.jsonl' })
  .option('seed',   { type: 'number', default: 42, describe: 'Random seed for reproducibility' })
  .argv;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function pad2(n) { return String(n).padStart(2, '0'); }

function pick(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function formatTime(h, m) {
  const hh = pad2(h);
  const mm = pad2(m);
  const standard = `${hh}:${mm}`;
  const r = Math.random();

  // 30% 12-hour format
  if (r < 0.30) {
    const period = h >= 12 ? 'PM' : 'AM';
    const h12 = h % 12 || 12;
    const sep = Math.random() > 0.5 ? ' ' : '';
    return `${h12}:${mm}${sep}${period}`;
  }
  // 10% spaced digits
  if (r < 0.40) return standard.split('').join(' ');
  // 5% dot separator
  if (r < 0.45) return `${hh}.${mm}`;
  // 5% no-pad hour
  if (r < 0.50) return `${h}:${mm}`;
  // 50% standard
  return standard;
}

/** Express a duration in natural language — varies phrasing */
function formatDuration(totalMinutes) {
  const abs = Math.abs(totalMinutes);
  const hours = Math.floor(abs / 60);
  const mins = abs % 60;

  // Special named durations
  if (abs === 15 && Math.random() < 0.3) return pick(['a quarter of an hour', 'quarter of an hour', '15 minutes', 'a quarter hour']);
  if (abs === 30 && Math.random() < 0.3) return pick(['half an hour', 'half hour', '30 minutes', 'thirty minutes']);
  if (abs === 45 && Math.random() < 0.3) return pick(['three quarters of an hour', 'forty-five minutes', '45 minutes', '45 min']);
  if (abs === 60 && Math.random() < 0.4) return pick(['an hour', 'one hour', '60 minutes', '1 hour']);
  if (abs === 90 && Math.random() < 0.3) return pick(['an hour and a half', 'one and a half hours', '90 minutes', 'ninety minutes']);
  if (abs === 120 && Math.random() < 0.3) return pick(['two hours', '2 hours', '120 minutes', 'a couple of hours']);

  // Hours + minutes
  if (hours > 0 && mins > 0) {
    const hPart = hours === 1 ? '1 hour' : `${hours} hours`;
    const mPart = mins === 1 ? '1 minute' : `${mins} minutes`;
    return Math.random() < 0.5 ? `${hPart} and ${mPart}` : `${hPart} ${mPart}`;
  }
  if (hours > 0 && mins === 0) {
    return hours === 1 ? pick(['an hour', '1 hour', '60 minutes']) : `${hours} hours`;
  }
  // Minutes only
  if (abs === 1) return pick(['1 minute', 'a minute', 'one minute']);
  if (abs === 5 && Math.random() < 0.3) return pick(['5 minutes', 'five minutes', '5 min']);
  if (abs === 10 && Math.random() < 0.3) return pick(['10 minutes', 'ten minutes', '10 min']);
  return Math.random() < 0.2 ? `${abs} min` : `${abs} minutes`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fuzzy Times
// ─────────────────────────────────────────────────────────────────────────────

const FUZZY_TIMES = [
  { spoken: "half past six",         h: 6,  m: 30 },
  { spoken: "quarter past one",      h: 13, m: 15 },
  { spoken: "quarter to ten",        h: 9,  m: 45 },
  { spoken: "noon",                  h: 12, m: 0  },
  { spoken: "midnight",              h: 0,  m: 0  },
  { spoken: "half past eight",       h: 8,  m: 30 },
  { spoken: "quarter past three",    h: 15, m: 15 },
  { spoken: "ten to five",           h: 16, m: 50 },
  { spoken: "twenty past seven",     h: 7,  m: 20 },
  { spoken: "five to nine",          h: 8,  m: 55 },
  { spoken: "half past eleven",      h: 23, m: 30 },
  { spoken: "quarter to midnight",   h: 23, m: 45 },
  // New colloquial times
  { spoken: "lunchtime",             h: 12, m: 30 },
  { spoken: "tea time",              h: 16, m: 0  },
  { spoken: "dawn",                  h: 6,  m: 0  },
  { spoken: "dusk",                  h: 18, m: 0  },
  { spoken: "crack of dawn",         h: 5,  m: 0  },
  { spoken: "happy hour",            h: 17, m: 0  },
  { spoken: "elevenses",             h: 11, m: 0  },
  { spoken: "supper time",           h: 19, m: 0  },
  { spoken: "closing time",          h: 23, m: 0  },
  { spoken: "first thing",           h: 8,  m: 0  },
  { spoken: "end of day",            h: 17, m: 0  },
  { spoken: "sunrise",               h: 6,  m: 30 },
  { spoken: "sunset",                h: 18, m: 30 },
  { spoken: "mid-morning",           h: 10, m: 0  },
  { spoken: "mid-afternoon",         h: 14, m: 30 },
  { spoken: "early evening",         h: 18, m: 0  },
  { spoken: "late morning",          h: 11, m: 30 },
];

// ─────────────────────────────────────────────────────────────────────────────
// Route Token Builder
// ─────────────────────────────────────────────────────────────────────────────

function routeTimeAdd(startH, startM, durH, durM) {
  const parts = ['[ROUTE]', '[ROUTE_TIME_ADD]',
    '[HEAD_TIME]', `[ARG_HOUR_${pad2(startH)}]`, `[ARG_MIN_${pad2(startM)}]`,
    '[HEAD_DURATION]'];
  if (durH > 0) parts.push(`[ARG_HOUR_${pad2(durH)}]`);
  parts.push(`[ARG_MIN_${pad2(durM)}]`, '[/ROUTE]');
  return parts.join(' ');
}

function routeTimeSub(startH, startM, durH, durM) {
  const parts = ['[ROUTE]', '[ROUTE_TIME_SUB]',
    '[HEAD_TIME]', `[ARG_HOUR_${pad2(startH)}]`, `[ARG_MIN_${pad2(startM)}]`,
    '[HEAD_DURATION]'];
  if (durH > 0) parts.push(`[ARG_HOUR_${pad2(durH)}]`);
  parts.push(`[ARG_MIN_${pad2(durM)}]`, '[/ROUTE]');
  return parts.join(' ');
}

function routeDurationBetween(h1, m1, h2, m2) {
  return [
    '[ROUTE]', '[ROUTE_DURATION_BETWEEN]',
    '[HEAD_TIME]', `[ARG_HOUR_${pad2(h1)}]`, `[ARG_MIN_${pad2(m1)}]`,
    '[HEAD_TIME]', `[ARG_HOUR_${pad2(h2)}]`, `[ARG_MIN_${pad2(m2)}]`,
    '[/ROUTE]'
  ].join(' ');
}

function noRoute(reason) {
  return `[NO_ROUTE] ${reason}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain Templates — 19 domains, 120+ templates
// Each template is a function(timeStr, durStr, direction) → question string
// direction: 'add' | 'sub'
// ─────────────────────────────────────────────────────────────────────────────

const TEMPLATES = {
  Professional: {
    add: [
      (t, d) => `The meeting starts at ${t} and lasts ${d}. When does it end?`,
      (t, d) => `Our standup is at ${t}. It usually runs for ${d}. What time will it finish?`,
      (t, d) => `I have a conference call at ${t} that will take ${d}. When can I schedule the next one?`,
      (t, d) => `My shift begins at ${t}. I get a break after ${d}. What time is my break?`,
      (t, d) => `The deadline is at ${t} but the client gave us an extension of ${d}. What's the new deadline?`,
      (t, d) => `The presentation starts at ${t} and is expected to run for ${d}. When should I book the room until?`,
      (t, d) => `We started the sprint review at ${t}. It's been going for ${d}. What time is it now?`,
      (t, d) => `Lunch break starts at ${t} and lasts ${d}. When do I need to be back?`,
      (t, d) => `The workshop kicks off at ${t} and runs for ${d}. What time does it wrap up?`,
      (t, d) => `I clocked in at ${t}. After working ${d} of overtime, what time did I finish?`,
    ],
    sub: [
      (t, d) => `The meeting ended at ${t}. It lasted ${d}. When did it start?`,
      (t, d) => `My shift ends at ${t}. I work for ${d}. When does my shift begin?`,
      (t, d) => `The deadline is at ${t}. I need ${d} to finish the work. When should I start?`,
      (t, d) => `The client call finished at ${t} after ${d}. What time did it begin?`,
      (t, d) => `I left the office at ${t}. I was in the meeting for ${d} before leaving. When did the meeting start?`,
      (t, d) => `The standup ended at ${t}. It ran for ${d}. What time did we kick off?`,
      (t, d) => `My break ends at ${t}. I've been on break for ${d}. When did my break start?`,
      (t, d) => `I finished my overtime at ${t}. I did ${d} of extra work. When did I start the overtime?`,
      (t, d) => `The presentation finished at ${t}. It was ${d} long. When did the presenter begin?`,
      (t, d) => `We wrapped up the sprint retro at ${t}. It took ${d}. When did we start?`,
    ],
    between: [
      (t1, t2) => `The meeting runs from ${t1} to ${t2}. How long is it?`,
      (t1, t2) => `My shift is ${t1} to ${t2}. How many minutes is that?`,
      (t1, t2) => `I worked from ${t1} until ${t2}. How long was my shift?`,
      (t1, t2) => `The workshop is scheduled from ${t1} to ${t2}. What's the duration?`,
    ],
  },

  Commuting: {
    add: [
      (t, d) => `I'm leaving home at ${t}. The commute takes ${d}. What time will I arrive?`,
      (t, d) => `The bus departs at ${t}. The journey is ${d}. When does it arrive?`,
      (t, d) => `I started walking at ${t}. It takes ${d} to reach the office. When will I get there?`,
      (t, d) => `My train leaves at ${t}. The ride is ${d}. What time does it reach the destination?`,
      (t, d) => `I ordered an Uber at ${t}. The ETA is ${d}. When will it arrive?`,
      (t, d) => `I got on my bike at ${t}. The cycle to work takes ${d}. What time will I arrive?`,
      (t, d) => `The flight departs at ${t}. The flight time is ${d}. When does it land?`,
      (t, d) => `I entered the car park at ${t}. The parking meter gives me ${d}. When does it expire?`,
      (t, d) => `Rush hour starts at ${t} and lasts ${d}. When does rush hour end?`,
      (t, d) => `I left the station at ${t}. The walk home takes ${d}. When do I get home?`,
    ],
    sub: [
      (t, d) => `I arrived at work at ${t}. The commute was ${d}. When did I leave home?`,
      (t, d) => `The bus arrived at ${t}. The journey was ${d}. When did it depart?`,
      (t, d) => `I got to the office at ${t} after walking for ${d}. When did I start walking?`,
      (t, d) => `My train arrived at ${t}. The journey took ${d}. When did it depart?`,
      (t, d) => `I need to arrive by ${t}. The drive takes ${d}. When should I leave?`,
      (t, d) => `The flight landed at ${t}. It was a ${d} flight. When did it take off?`,
      (t, d) => `My parking expired at ${t}. I paid for ${d}. When did I park?`,
      (t, d) => `I got home at ${t}. The walk from the station was ${d}. When did I leave the station?`,
      (t, d) => `Rush hour ended at ${t}. It lasted ${d}. When did it start?`,
      (t, d) => `The Uber arrived at ${t} after a ${d} wait. When did I request it?`,
    ],
    between: [
      (t1, t2) => `I left at ${t1} and arrived at ${t2}. How long was my commute?`,
      (t1, t2) => `The bus departed at ${t1} and arrived at ${t2}. How long was the journey?`,
      (t1, t2) => `I started driving at ${t1} and parked at ${t2}. How long did I drive?`,
    ],
  },

  Domestic: {
    add: [
      (t, d) => `I put the roast in the oven at ${t}. It needs ${d}. When is it done?`,
      (t, d) => `I started the washing machine at ${t}. The cycle takes ${d}. When will it finish?`,
      (t, d) => `The dishwasher started at ${t} and runs for ${d}. When does it finish?`,
      (t, d) => `I started watering the garden at ${t}. I water for ${d}. When will I finish?`,
      (t, d) => `I fed the cat at ${t}. The next feed is in ${d}. When do I feed it again?`,
      (t, d) => `The delivery window starts at ${t}. They said to allow ${d}. When does the window close?`,
      (t, d) => `I put the bread in the oven at ${t}. It bakes for ${d}. When should I take it out?`,
      (t, d) => `I turned on the sprinklers at ${t}. They run for ${d}. When do they turn off?`,
    ],
    sub: [
      (t, d) => `The roast was done at ${t}. It cooked for ${d}. When did it go in?`,
      (t, d) => `The washing machine finished at ${t}. The cycle was ${d}. When did it start?`,
      (t, d) => `The dishwasher beeped at ${t}. It ran for ${d}. When did I start it?`,
      (t, d) => `I need to serve dinner at ${t}. The recipe takes ${d}. When should I start cooking?`,
      (t, d) => `The cat needs feeding at ${t}. The last feed was ${d} ago. When was the last feed?`,
      (t, d) => `The delivery arrived at ${t}. The estimated wait was ${d}. When did they say to expect it?`,
      (t, d) => `The bread was ready at ${t}. It baked for ${d}. When did I put it in?`,
      (t, d) => `The sprinklers turned off at ${t}. They ran for ${d}. When did they start?`,
    ],
    between: [
      (t1, t2) => `I started cooking at ${t1} and finished at ${t2}. How long did it take?`,
      (t1, t2) => `The laundry started at ${t1} and finished at ${t2}. How long was the cycle?`,
    ],
  },

  Social: {
    add: [
      (t, d) => `My dinner reservation is at ${t}. I arrived ${d} late. What time did I arrive?`,
      (t, d) => `The party starts at ${t}. I'll be there ${d} after it starts. What time will I arrive?`,
      (t, d) => `We're meeting at the pub at ${t}. I'll be ${d} late. When will I show up?`,
      (t, d) => `The phone call started at ${t} and lasted ${d}. When did it end?`,
      (t, d) => `Our video call is at ${t}. I expect it to go on for ${d}. When will it end?`,
      (t, d) => `I'm meeting a friend at ${t}. My friend texted they'll be ${d} late. When do I actually expect them?`,
      (t, d) => `The birthday party begins at ${t} and we've booked the venue for ${d}. When must we leave?`,
      (t, d) => `Date night starts at ${t}. We're planning to stay out for ${d}. When will we head home?`,
    ],
    sub: [
      (t, d) => `My dinner reservation is at ${t}. I arrived ${d} early. What time did I arrive?`,
      (t, d) => `The party is at ${t}. I want to arrive ${d} early to help set up. When should I get there?`,
      (t, d) => `The pub quiz starts at ${t}. I need to be there ${d} before to grab a table. What time should I arrive?`,
      (t, d) => `The phone call ended at ${t}. We talked for ${d}. When did the call start?`,
      (t, d) => `Our video call ended at ${t} after ${d}. What time did it start?`,
      (t, d) => `My friend arrived at ${t} but they were ${d} late. What time were they supposed to arrive?`,
      (t, d) => `We left the venue at ${t}. The party was ${d} long. When did it start?`,
      (t, d) => `Date night ended at ${t}. We were out for ${d}. When did it start?`,
    ],
    between: [
      (t1, t2) => `The phone call lasted from ${t1} to ${t2}. How long did we talk?`,
      (t1, t2) => `We were at the pub from ${t1} to ${t2}. How long were we there?`,
      (t1, t2) => `The party ran from ${t1} until ${t2}. How long did it go?`,
    ],
  },

  Health: {
    add: [
      (t, d) => `Take your medication at ${t}. The next dose is in ${d}. When is the next dose?`,
      (t, d) => `I started my workout at ${t}. It takes ${d}. When will I finish?`,
      (t, d) => `I went to bed at ${t}. I want to sleep for ${d}. When should my alarm go off?`,
      (t, d) => `My therapy session starts at ${t} and lasts ${d}. When does it end?`,
      (t, d) => `I started fasting at ${t}. I'm fasting for ${d}. When can I eat again?`,
      (t, d) => `The doctor's appointment is at ${t}. Allow ${d} for the consultation. When will I be done?`,
      (t, d) => `I started my physiotherapy exercises at ${t}. The routine takes ${d}. When will I finish?`,
      (t, d) => `My meditation session starts at ${t} and lasts ${d}. When does it end?`,
    ],
    sub: [
      (t, d) => `My next dose is at ${t}. I take it every ${d}. When was the previous dose?`,
      (t, d) => `I finished my workout at ${t}. It took ${d}. When did I start?`,
      (t, d) => `My alarm went off at ${t}. I slept for ${d}. When did I go to bed?`,
      (t, d) => `My therapy session ended at ${t}. It lasted ${d}. When did it start?`,
      (t, d) => `I'm breaking my fast at ${t}. I fasted for ${d}. When did I start fasting?`,
      (t, d) => `The appointment finished at ${t}. The consultation took ${d}. When did it start?`,
      (t, d) => `I finished my exercises at ${t}. The routine was ${d}. When did I begin?`,
      (t, d) => `I need to take my medication at ${t}. I need to eat ${d} before taking it. When should I eat?`,
    ],
    between: [
      (t1, t2) => `I went to bed at ${t1} and woke up at ${t2}. How long did I sleep?`,
      (t1, t2) => `The workout started at ${t1} and ended at ${t2}. How long was it?`,
      (t1, t2) => `My fast was from ${t1} to ${t2}. How long did I fast?`,
    ],
  },

  Education: {
    add: [
      (t, d) => `The lecture starts at ${t} and lasts ${d}. When does it end?`,
      (t, d) => `The exam begins at ${t}. You have ${d} to complete it. When does it end?`,
      (t, d) => `I started studying at ${t}. I plan to study for ${d}. When will I stop?`,
      (t, d) => `Office hours start at ${t} and run for ${d}. When do they end?`,
      (t, d) => `The tutorial begins at ${t} and is ${d} long. When does it finish?`,
      (t, d) => `The library opens at ${t}. I'll study for ${d}. When will I finish?`,
      (t, d) => `The school day starts at ${t}. The first period is ${d}. When does it end?`,
      (t, d) => `The homework deadline is at ${t} but the teacher gave a ${d} extension. What's the new deadline?`,
    ],
    sub: [
      (t, d) => `The lecture ended at ${t}. It lasted ${d}. When did it start?`,
      (t, d) => `The exam finished at ${t}. It was ${d} long. When did it begin?`,
      (t, d) => `I finished studying at ${t} after ${d}. When did I start?`,
      (t, d) => `Office hours end at ${t}. They run for ${d}. When do they start?`,
      (t, d) => `The tutorial finished at ${t}. It was ${d}. When did it begin?`,
      (t, d) => `The library closes at ${t}. I need ${d} to study. When should I arrive?`,
      (t, d) => `School pickup is at ${t}. The last class is ${d}. When does the last class start?`,
      (t, d) => `The homework is due at ${t}. It will take me ${d} to finish. When should I start?`,
    ],
    between: [
      (t1, t2) => `The lecture was from ${t1} to ${t2}. How long was it?`,
      (t1, t2) => `I studied from ${t1} to ${t2}. How long did I study?`,
      (t1, t2) => `The exam ran from ${t1} to ${t2}. What was the duration?`,
    ],
  },

  Entertainment: {
    add: [
      (t, d) => `The movie starts at ${t} and runs for ${d}. When does it end?`,
      (t, d) => `The concert begins at ${t}. The set is ${d}. When does it finish?`,
      (t, d) => `The TV episode starts at ${t} and is ${d} long. When does it end?`,
      (t, d) => `I started the podcast at ${t}. It's ${d} long. When will it finish?`,
      (t, d) => `The gaming session started at ${t}. We've been playing for ${d}. What time is it now?`,
      (t, d) => `The play starts at ${t}. Act one is ${d}. When is the intermission?`,
      (t, d) => `The stream goes live at ${t} and runs for ${d}. When does it end?`,
      (t, d) => `The match kicks off at ${t}. Each half is ${d}. When does the first half end?`,
    ],
    sub: [
      (t, d) => `The movie ended at ${t}. It was ${d} long. When did it start?`,
      (t, d) => `The concert finished at ${t}. The set was ${d}. When did it begin?`,
      (t, d) => `The episode ended at ${t}. It was ${d}. When did it start?`,
      (t, d) => `The podcast finished at ${t} and was ${d} long. When did I start it?`,
      (t, d) => `I stopped gaming at ${t} after ${d}. When did I start?`,
      (t, d) => `The intermission started at ${t}. Act one was ${d}. When did the play start?`,
      (t, d) => `The stream ended at ${t} after ${d}. When did it go live?`,
      (t, d) => `The first half ended at ${t}. Each half is ${d}. When did the match kick off?`,
    ],
    between: [
      (t1, t2) => `The movie ran from ${t1} to ${t2}. How long was it?`,
      (t1, t2) => `I gamed from ${t1} to ${t2}. How long did I play?`,
      (t1, t2) => `The concert was from ${t1} to ${t2}. How long was it?`,
    ],
  },

  Parenting: {
    add: [
      (t, d) => `Baby fell asleep at ${t}. Naps usually last ${d}. When will they wake up?`,
      (t, d) => `School starts at ${t}. The first class is ${d}. When does it end?`,
      (t, d) => `Bedtime is at ${t}. The bedtime story takes ${d}. When will the child be asleep?`,
      (t, d) => `Bath time starts at ${t} and lasts ${d}. When does it end?`,
      (t, d) => `I started feeding the baby at ${t}. Feeding takes ${d}. When will it finish?`,
      (t, d) => `Playtime started at ${t} and lasts ${d}. When is it over?`,
    ],
    sub: [
      (t, d) => `Baby woke up at ${t} after a ${d} nap. When did they fall asleep?`,
      (t, d) => `I need to pick up my child at ${t}. The drive takes ${d}. When should I leave?`,
      (t, d) => `The child fell asleep at ${t}. The bedtime routine took ${d}. When did it start?`,
      (t, d) => `Bath time ended at ${t}. It lasted ${d}. When did it start?`,
      (t, d) => `The baby finished feeding at ${t} after ${d}. When did feeding start?`,
      (t, d) => `Playtime ended at ${t}. We played for ${d}. When did it start?`,
    ],
    between: [
      (t1, t2) => `The baby napped from ${t1} to ${t2}. How long was the nap?`,
      (t1, t2) => `School runs from ${t1} to ${t2}. How long is the school day?`,
    ],
  },

  Sports: {
    add: [
      (t, d) => `The match kicks off at ${t}. The first half is ${d}. When is halftime?`,
      (t, d) => `Practice starts at ${t} and runs for ${d}. When does it end?`,
      (t, d) => `Warm-up begins at ${t} and lasts ${d}. When does the main event start?`,
      (t, d) => `The race starts at ${t}. My expected time is ${d}. When should I cross the finish line?`,
      (t, d) => `My gym session started at ${t}. I work out for ${d}. When will I finish?`,
      (t, d) => `The cool-down period starts at ${t} and lasts ${d}. When does it end?`,
    ],
    sub: [
      (t, d) => `Halftime started at ${t}. The first half was ${d}. When did the match kick off?`,
      (t, d) => `Practice ended at ${t}. It was ${d} long. When did it start?`,
      (t, d) => `The main event starts at ${t}. Warm-up is ${d}. When should warm-up begin?`,
      (t, d) => `I finished my run at ${t}. It took ${d}. When did I start?`,
      (t, d) => `I left the gym at ${t} after a ${d} session. When did I start?`,
      (t, d) => `The match ended at ${t}. Each half was ${d}. When did the second half start?`,
    ],
    between: [
      (t1, t2) => `The match ran from ${t1} to ${t2}. How long was it?`,
      (t1, t2) => `I was at the gym from ${t1} to ${t2}. How long was my session?`,
    ],
  },

  Cooking: {
    add: [
      (t, d) => `I put the chicken in the marinade at ${t}. It needs to marinate for ${d}. When is it ready?`,
      (t, d) => `The cake went into the oven at ${t}. It bakes for ${d}. When should I take it out?`,
      (t, d) => `I started proofing the dough at ${t}. It needs ${d}. When will it be ready?`,
      (t, d) => `The steak came off the grill at ${t}. It needs to rest for ${d}. When can I serve it?`,
      (t, d) => `I put the water on to boil at ${t}. It takes ${d} to boil. When will it be ready?`,
      (t, d) => `The slow cooker went on at ${t}. It needs to simmer for ${d}. When is dinner ready?`,
    ],
    sub: [
      (t, d) => `The chicken is ready at ${t}. It marinates for ${d}. When should I put it in the marinade?`,
      (t, d) => `The cake needs to come out at ${t}. It bakes for ${d}. When should I put it in?`,
      (t, d) => `I need the dough ready by ${t}. Proofing takes ${d}. When should I start?`,
      (t, d) => `I need to serve the steak at ${t}. It needs ${d} to rest. When should I take it off the grill?`,
      (t, d) => `Dinner is at ${t}. The slow cooker needs ${d}. When should I turn it on?`,
      (t, d) => `I want the pasta ready by ${t}. Cooking takes ${d}. When should I start?`,
    ],
    between: [
      (t1, t2) => `The cake was in the oven from ${t1} to ${t2}. How long did it bake?`,
      (t1, t2) => `The stew simmered from ${t1} to ${t2}. How long was it on?`,
    ],
  },

  Financial: {
    add: [
      (t, d) => `The market opens at ${t}. I plan to check prices after ${d}. What time is that?`,
      (t, d) => `The auction starts at ${t}. My lot comes up after ${d}. When does bidding open for me?`,
      (t, d) => `Trading opens at ${t}. The window is ${d}. When does trading close?`,
      (t, d) => `The payment is due at ${t}. They gave a grace period of ${d}. What's the final deadline?`,
    ],
    sub: [
      (t, d) => `I checked the market at ${t}. The market opened ${d} before that. When did it open?`,
      (t, d) => `The auction ends at ${t}. It runs for ${d}. When did it start?`,
      (t, d) => `Trading closes at ${t}. The window is ${d}. When did trading open?`,
      (t, d) => `The invoice was due at ${t}. I was notified ${d} in advance. When was I notified?`,
    ],
    between: [
      (t1, t2) => `The market was open from ${t1} to ${t2}. How long was the trading day?`,
    ],
  },

  Travel: {
    add: [
      (t, d) => `Check-in opens at ${t}. Allow ${d} for the process. When should I expect to be done?`,
      (t, d) => `My hotel checkout is at ${t}. I have a ${d} transfer to the airport. When do I arrive at the airport?`,
      (t, d) => `The guided tour starts at ${t} and lasts ${d}. When does it end?`,
      (t, d) => `The ferry departs at ${t}. The crossing takes ${d}. When does it arrive?`,
      (t, d) => `My layover starts at ${t}. It's ${d} long. When does my connecting flight depart?`,
      (t, d) => `The excursion departs at ${t} and runs for ${d}. When does it return?`,
    ],
    sub: [
      (t, d) => `Check-in closes at ${t}. Allow ${d} for check-in. When should I arrive?`,
      (t, d) => `My transfer arrives at the airport at ${t}. The journey takes ${d}. When did it leave the hotel?`,
      (t, d) => `The tour ended at ${t}. It was ${d}. When did it start?`,
      (t, d) => `The ferry arrived at ${t}. The crossing was ${d}. When did it depart?`,
      (t, d) => `My connecting flight departs at ${t}. I have a ${d} layover. When did I arrive?`,
      (t, d) => `The excursion returned at ${t}. It ran for ${d}. When did it depart?`,
    ],
    between: [
      (t1, t2) => `The ferry sailed from ${t1} to ${t2}. How long was the crossing?`,
      (t1, t2) => `The tour ran from ${t1} to ${t2}. How long was it?`,
    ],
  },

  Tech: {
    add: [
      (t, d) => `Deployment started at ${t}. The pipeline takes ${d}. When does it go live?`,
      (t, d) => `The CI build started at ${t}. It usually takes ${d}. When should it finish?`,
      (t, d) => `The incident started at ${t}. The SLA response time is ${d}. When is the SLA deadline?`,
      (t, d) => `The server rebooted at ${t}. Boot time is ${d}. When will it be back online?`,
    ],
    sub: [
      (t, d) => `The site went live at ${t}. Deployment took ${d}. When did deployment start?`,
      (t, d) => `The build finished at ${t}. It took ${d}. When did it start?`,
      (t, d) => `The incident was resolved at ${t}. Total downtime was ${d}. When did it start?`,
      (t, d) => `The server came back online at ${t}. Boot time was ${d}. When did it go down?`,
    ],
    between: [
      (t1, t2) => `The outage was from ${t1} to ${t2}. How long was the downtime?`,
      (t1, t2) => `The deploy ran from ${t1} to ${t2}. How long did it take?`,
    ],
  },

  Emergency: {
    add: [
      (t, d) => `The ambulance was dispatched at ${t}. The estimated arrival is ${d}. When will it arrive?`,
      (t, d) => `The observation period started at ${t}. The patient needs ${d} of monitoring. When does it end?`,
      (t, d) => `The night shift starts at ${t} and is ${d} long. When does it end?`,
      (t, d) => `The emergency call came in at ${t}. Response time is ${d}. When will the team arrive?`,
    ],
    sub: [
      (t, d) => `The ambulance arrived at ${t}. It took ${d} to reach the scene. When was it dispatched?`,
      (t, d) => `The observation period ends at ${t}. It lasts ${d}. When did it start?`,
      (t, d) => `The night shift ends at ${t}. It's ${d} long. When did it start?`,
      (t, d) => `The response team arrived at ${t}. Response time was ${d}. When was the call?`,
    ],
    between: [
      (t1, t2) => `The shift ran from ${t1} to ${t2}. How long was it?`,
    ],
  },

  Agriculture: {
    add: [
      (t, d) => `Irrigation starts at ${t} and runs for ${d}. When does it stop?`,
      (t, d) => `The harvest crew arrives at ${t}. They'll work for ${d}. When do they finish?`,
      (t, d) => `I started the livestock feed at ${t}. It takes ${d}. When will I finish?`,
      (t, d) => `Spraying starts at ${t} and takes ${d}. When does it end?`,
    ],
    sub: [
      (t, d) => `Irrigation ends at ${t}. It runs for ${d}. When does it start?`,
      (t, d) => `The harvest crew finished at ${t}. They worked for ${d}. When did they start?`,
      (t, d) => `I finished feeding the livestock at ${t}. It took ${d}. When did I start?`,
      (t, d) => `Spraying finished at ${t}. It took ${d}. When did it start?`,
    ],
    between: [
      (t1, t2) => `The harvest ran from ${t1} to ${t2}. How long did it take?`,
    ],
  },

  Creative: {
    add: [
      (t, d) => `The studio is booked from ${t} for ${d}. When does the booking end?`,
      (t, d) => `Rehearsal starts at ${t} and runs for ${d}. When does it finish?`,
      (t, d) => `The gallery opening starts at ${t}. The event is ${d}. When does it end?`,
      (t, d) => `The recording session begins at ${t} and is booked for ${d}. When does it wrap?`,
    ],
    sub: [
      (t, d) => `The studio booking ends at ${t}. It was booked for ${d}. When did it start?`,
      (t, d) => `Rehearsal finished at ${t}. It ran for ${d}. When did it start?`,
      (t, d) => `The gallery event ended at ${t}. It lasted ${d}. When did it open?`,
      (t, d) => `The recording wrapped at ${t}. The session was ${d}. When did it begin?`,
    ],
    between: [
      (t1, t2) => `The rehearsal ran from ${t1} to ${t2}. How long was it?`,
    ],
  },

  Maintenance: {
    add: [
      (t, d) => `The mechanic started at ${t}. The repair takes ${d}. When will it be done?`,
      (t, d) => `The plumber arrived at ${t}. The job will take ${d}. When will they finish?`,
      (t, d) => `The electrician started work at ${t}. It takes ${d}. When will they be done?`,
      (t, d) => `The service appointment is at ${t}. Allow ${d} for the work. When will it be done?`,
    ],
    sub: [
      (t, d) => `The repair was finished at ${t}. It took ${d}. When did the mechanic start?`,
      (t, d) => `The plumber finished at ${t}. The job took ${d}. When did they start?`,
      (t, d) => `The electrician finished at ${t} after ${d}. When did they start?`,
      (t, d) => `The service was completed at ${t}. It took ${d}. When was the appointment?`,
    ],
    between: [
      (t1, t2) => `The repair ran from ${t1} to ${t2}. How long did it take?`,
    ],
  },

  History: {
    add: [
      (t, d) => `The ceremony began at ${t} and lasted ${d}. When did it end?`,
      (t, d) => `The march started at ${t} and went on for ${d}. When did it finish?`,
      (t, d) => `The battle commenced at ${t}. The engagement lasted ${d}. When did it end?`,
      (t, d) => `The speech began at ${t} and ran for ${d}. When did it conclude?`,
    ],
    sub: [
      (t, d) => `The ceremony ended at ${t}. It lasted ${d}. When did it begin?`,
      (t, d) => `The march ended at ${t}. It went on for ${d}. When did it start?`,
      (t, d) => `The battle ended at ${t}. The engagement was ${d}. When did it begin?`,
      (t, d) => `The speech ended at ${t} after ${d}. When did it start?`,
    ],
    between: [
      (t1, t2) => `The siege lasted from ${t1} to ${t2}. How long was it?`,
    ],
  },

  Procrastination: {
    add: [
      (t, d) => `I planned to start working at ${t} but procrastinated for ${d}. When did I actually start?`,
      (t, d) => `I hit snooze at ${t}. I snoozed for ${d}. When did I actually get up?`,
      (t, d) => `I said "5 more minutes" at ${t}. But I actually waited ${d}. When did I start?`,
    ],
    sub: [
      (t, d) => `I actually started working at ${t}. I procrastinated for ${d}. When did I plan to start?`,
      (t, d) => `I finally got up at ${t}. I snoozed for ${d}. When did I first set my alarm?`,
      (t, d) => `I started the task at ${t}. I was ${d} late. What was my original start time?`,
    ],
    between: [
      (t1, t2) => `I planned to start at ${t1} but didn't start until ${t2}. How long did I procrastinate?`,
    ],
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// NO_ROUTE Templates
// ─────────────────────────────────────────────────────────────────────────────

const NO_ROUTE_TEMPLATES = [
  // Base-10 arithmetic
  () => { const a = faker.number.int({min:1,max:500}); const b = faker.number.int({min:1,max:500}); return { q: `What is ${a} + ${b}?`, r: 'This is base-10 arithmetic, not temporal.' }; },
  () => { const a = faker.number.int({min:1,max:100}); const b = faker.number.int({min:1,max:a}); return { q: `What is ${a} - ${b}?`, r: 'This is base-10 arithmetic, not temporal.' }; },
  () => { const a = faker.number.int({min:2,max:20}); const b = faker.number.int({min:2,max:20}); return { q: `What is ${a} × ${b}?`, r: 'This is multiplication, not a time question.' }; },
  () => { const pct = pick([10,15,20,25]); const v = faker.number.int({min:50,max:500}); return { q: `Calculate ${pct}% of ${v}.`, r: 'This is a percentage calculation, not temporal.' }; },
  // Non-temporal questions
  () => ({ q: `How many people were at the meeting?`, r: 'This is a count question, not a time calculation.' }),
  () => ({ q: `What colour is the sky?`, r: 'This is not a temporal question.' }),
  () => ({ q: `Who won the match last night?`, r: 'This is a factual question, not a time calculation.' }),
  () => ({ q: `What's the weather like today?`, r: 'This is a weather question, not temporal.' }),
  () => ({ q: `How far is it from London to Manchester?`, r: 'This is a distance question, not temporal.' }),
  () => ({ q: `Convert 5 miles to kilometres.`, r: 'This is a unit conversion, not a time calculation.' }),
  () => ({ q: `What's the capital of France?`, r: 'This is a geography question, not temporal.' }),
  () => ({ q: `How many days are in February?`, r: 'This is a calendar fact, not a time computation.' }),
  // Time-adjacent but non-computable
  () => ({ q: `Is 3pm a good time for a meeting?`, r: 'This is an opinion, not a temporal computation.' }),
  () => ({ q: `Do you prefer morning or evening workouts?`, r: 'This is a preference question, not temporal.' }),
  () => ({ q: `What time do most people wake up?`, r: 'This is a general knowledge question, not a specific time computation.' }),
  () => ({ q: `Is it too late to call someone at 10pm?`, r: 'This is a social judgment, not a time calculation.' }),
  () => ({ q: `What's a good bedtime for a 5-year-old?`, r: 'This is parenting advice, not a time computation.' }),
  () => ({ q: `Should I eat before or after my workout?`, r: 'This is health advice, not temporal.' }),
  // Temporal words but no computation
  () => { const t = formatTime(faker.number.int({min:0,max:23}), faker.number.int({min:0,max:59})); return { q: `Remind me at ${t} to take out the bins.`, r: 'This is a reminder request, not a time computation.' }; },
  () => { const t = formatTime(faker.number.int({min:0,max:23}), faker.number.int({min:0,max:59})); return { q: `Set an alarm for ${t}.`, r: 'This is an alarm request, not a time computation.' }; },
];

// ─────────────────────────────────────────────────────────────────────────────
// Record Generation
// ─────────────────────────────────────────────────────────────────────────────

const DOMAIN_NAMES = Object.keys(TEMPLATES);

function generateAddRecord() {
  const domain = pick(DOMAIN_NAMES);
  const templates = TEMPLATES[domain].add;
  const template = pick(templates);

  const useFuzzy = Math.random() < 0.12;
  let startH, startM, timeStr;

  if (useFuzzy) {
    const fuzzy = pick(FUZZY_TIMES);
    startH = fuzzy.h;
    startM = fuzzy.m;
    timeStr = fuzzy.spoken;
  } else {
    startH = faker.number.int({ min: 0, max: 23 });
    startM = faker.number.int({ min: 0, max: 59 });
    timeStr = formatTime(startH, startM);
  }

  const totalDelta = faker.number.int({ min: 1, max: 300 });
  const durH = Math.floor(totalDelta / 60);
  const durM = totalDelta % 60;
  const durStr = formatDuration(totalDelta);

  const question = template(timeStr, durStr);
  const route = routeTimeAdd(startH, startM, durH, durM);

  return { text: `${question}\n${route}` };
}

function generateSubRecord() {
  const domain = pick(DOMAIN_NAMES);
  const templates = TEMPLATES[domain].sub;
  const template = pick(templates);

  const useFuzzy = Math.random() < 0.12;
  let startH, startM, timeStr;

  if (useFuzzy) {
    const fuzzy = pick(FUZZY_TIMES);
    startH = fuzzy.h;
    startM = fuzzy.m;
    timeStr = fuzzy.spoken;
  } else {
    startH = faker.number.int({ min: 0, max: 23 });
    startM = faker.number.int({ min: 0, max: 59 });
    timeStr = formatTime(startH, startM);
  }

  const totalDelta = faker.number.int({ min: 1, max: 300 });
  const durH = Math.floor(totalDelta / 60);
  const durM = totalDelta % 60;
  const durStr = formatDuration(totalDelta);

  const question = template(timeStr, durStr);
  const route = routeTimeSub(startH, startM, durH, durM);

  return { text: `${question}\n${route}` };
}

function generateBetweenRecord() {
  const domain = pick(DOMAIN_NAMES);
  const templates = TEMPLATES[domain].between;
  if (!templates || templates.length === 0) return generateAddRecord(); // fallback
  const template = pick(templates);

  const h1 = faker.number.int({ min: 0, max: 20 });
  const m1 = faker.number.int({ min: 0, max: 59 });
  // Ensure t2 > t1
  const gapMinutes = faker.number.int({ min: 10, max: 300 });
  const total2 = h1 * 60 + m1 + gapMinutes;
  const h2 = Math.floor(total2 / 60) % 24;
  const m2 = total2 % 60;

  const t1Str = formatTime(h1, m1);
  const t2Str = formatTime(h2, m2);

  const question = template(t1Str, t2Str);
  const route = routeDurationBetween(h1, m1, h2, m2);

  return { text: `${question}\n${route}` };
}

function generateNoRouteRecord() {
  const template = pick(NO_ROUTE_TEMPLATES);
  const { q, r } = template();
  return { text: `${q}\n${noRoute(r)}` };
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

async function main() {
  const trainStream = fs.createWriteStream(argv.output);
  const evalStream = fs.createWriteStream(argv.eval);
  const evalThreshold = 0.05; // 5% eval split

  // Distribution: 35% ADD, 35% SUB, 15% DURATION_BETWEEN, 15% NO_ROUTE
  const generators = [
    { fn: generateAddRecord, weight: 0.35 },
    { fn: generateSubRecord, weight: 0.70 },        // cumulative
    { fn: generateBetweenRecord, weight: 0.85 },     // cumulative
    { fn: generateNoRouteRecord, weight: 1.00 },      // cumulative
  ];

  const stats = { add: 0, sub: 0, between: 0, noroute: 0, train: 0, eval: 0 };

  for (let i = 0; i < argv.count; i++) {
    const roll = Math.random();
    let record;

    if (roll < generators[0].weight) {
      record = generateAddRecord();
      stats.add++;
    } else if (roll < generators[1].weight) {
      record = generateSubRecord();
      stats.sub++;
    } else if (roll < generators[2].weight) {
      record = generateBetweenRecord();
      stats.between++;
    } else {
      record = generateNoRouteRecord();
      stats.noroute++;
    }

    const isEval = Math.random() < evalThreshold;
    const stream = isEval ? evalStream : trainStream;
    stream.write(JSON.stringify({ text: record.text }) + '\n');

    if (isEval) stats.eval++;
    else stats.train++;

    if ((i + 1) % 10000 === 0) {
      console.log(`  Generated ${i + 1}/${argv.count} records...`);
    }
  }

  trainStream.end();
  evalStream.end();

  console.log(`\n✅ Generation complete!`);
  console.log(`   Total:   ${argv.count}`);
  console.log(`   Train:   ${stats.train} → ${argv.output}`);
  console.log(`   Eval:    ${stats.eval} → ${argv.eval}`);
  console.log(`\n   Distribution:`);
  console.log(`   TIME_ADD:          ${stats.add} (${(stats.add/argv.count*100).toFixed(1)}%)`);
  console.log(`   TIME_SUB:          ${stats.sub} (${(stats.sub/argv.count*100).toFixed(1)}%)`);
  console.log(`   DURATION_BETWEEN:  ${stats.between} (${(stats.between/argv.count*100).toFixed(1)}%)`);
  console.log(`   NO_ROUTE:          ${stats.noroute} (${(stats.noroute/argv.count*100).toFixed(1)}%)`);
}

main().catch(console.error);
