/**
 * Tests for the Temporal Computation Engine
 * ==========================================
 * Validates all routing operations using Luxon deterministic computation.
 */

const {
  parseTimeTokens,
  parseDurationTokens,
  parseRoutingCall,
  toTimeTokens,
  timeAdd,
  timeSub,
  durationBetween,
  calendarShift,
  timezoneConvert,
  execute,
} = require('./temporal_engine');

let passed = 0;
let failed = 0;

function assert(condition, name) {
  if (condition) {
    console.log(`  ✓ ${name}`);
    passed++;
  } else {
    console.log(`  ✗ ${name}`);
    failed++;
  }
}

function eq(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

// ---------------------------------------------------------------------------
// Token Parsing
// ---------------------------------------------------------------------------
console.log('\n=== Token Parsing ===');

const time = parseTimeTokens(['[HEAD_TIME]', '[ARG_HOUR_14]', '[ARG_MIN_20]']);
assert(time.hour === 14 && time.minute === 20, 'parseTimeTokens: 14:20');

const dur = parseDurationTokens(['[HEAD_DURATION]', '[ARG_HOUR_01]', '[ARG_MIN_30]']);
assert(dur.hours === 1 && dur.minutes === 30, 'parseDurationTokens: 1h30m');

const tokens = toTimeTokens({ hour: 9, minute: 3 });
assert(eq(tokens, ['[HEAD_TIME]', '[ARG_HOUR_09]', '[ARG_MIN_03]']), 'toTimeTokens: 09:03');

// ---------------------------------------------------------------------------
// Routing Call Parser
// ---------------------------------------------------------------------------
console.log('\n=== Routing Call Parser ===');

const call1 = parseRoutingCall(
  '[ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_09] [ARG_MIN_58] [HEAD_DURATION] [ARG_MIN_05]'
);
assert(call1.operation === '[ROUTE_TIME_ADD]', 'parseRoutingCall: operation');
assert(call1.operands.length === 2, 'parseRoutingCall: 2 operands');
assert(call1.operands[0][0] === '[HEAD_TIME]', 'parseRoutingCall: first operand is time');
assert(call1.operands[1][0] === '[HEAD_DURATION]', 'parseRoutingCall: second operand is duration');

// ---------------------------------------------------------------------------
// ROUTE_TIME_ADD — Arithmetic Mode
// ---------------------------------------------------------------------------
console.log('\n=== ROUTE_TIME_ADD ===');

// Simple addition: 09:58 + 5m = 10:03 (minute carry)
const add1 = timeAdd(
  ['[HEAD_TIME]', '[ARG_HOUR_09]', '[ARG_MIN_58]'],
  ['[HEAD_DURATION]', '[ARG_MIN_05]']
);
assert(add1.resultString === '10:03', 'Simple carry: 09:58 + 5m = 10:03');

// Hour rollover: 23:45 + 30m = 00:15
const add2 = timeAdd(
  ['[HEAD_TIME]', '[ARG_HOUR_23]', '[ARG_MIN_45]'],
  ['[HEAD_DURATION]', '[ARG_MIN_30]']
);
assert(add2.resultString === '00:15', 'Hour rollover: 23:45 + 30m = 00:15');
assert(add2.dayOverflow === true, 'Day overflow flag set');

// No carry: 08:30 + 15m = 08:45
const add3 = timeAdd(
  ['[HEAD_TIME]', '[ARG_HOUR_08]', '[ARG_MIN_30]'],
  ['[HEAD_DURATION]', '[ARG_MIN_15]']
);
assert(add3.resultString === '08:45', 'No carry: 08:30 + 15m = 08:45');

// Multi-hour: 14:00 + 2h30m = 16:30
const add4 = timeAdd(
  ['[HEAD_TIME]', '[ARG_HOUR_14]', '[ARG_MIN_00]'],
  ['[HEAD_DURATION]', '[ARG_HOUR_02]', '[ARG_MIN_30]']
);
assert(add4.resultString === '16:30', 'Multi-hour: 14:00 + 2h30m = 16:30');

// Cascade: 23:59 + 2m = 00:01
const add5 = timeAdd(
  ['[HEAD_TIME]', '[ARG_HOUR_23]', '[ARG_MIN_59]'],
  ['[HEAD_DURATION]', '[ARG_MIN_02]']
);
assert(add5.resultString === '00:01', 'Cascade: 23:59 + 2m = 00:01');

// ---------------------------------------------------------------------------
// ROUTE_TIME_SUB — Backwards Borrow
// ---------------------------------------------------------------------------
console.log('\n=== ROUTE_TIME_SUB ===');

// Simple subtraction: 10:00 - 15m = 09:45
const sub1 = timeSub(
  ['[HEAD_TIME]', '[ARG_HOUR_10]', '[ARG_MIN_00]'],
  ['[HEAD_DURATION]', '[ARG_MIN_15]']
);
assert(sub1.resultString === '09:45', 'Simple borrow: 10:00 - 15m = 09:45');

// Day underflow: 00:05 - 10m = 23:55
const sub2 = timeSub(
  ['[HEAD_TIME]', '[ARG_HOUR_00]', '[ARG_MIN_05]'],
  ['[HEAD_DURATION]', '[ARG_MIN_10]']
);
assert(sub2.resultString === '23:55', 'Day underflow: 00:05 - 10m = 23:55');
assert(sub2.dayUnderflow === true, 'Day underflow flag set');

// Multi-hour: 12:00 - 2h = 10:00
const sub3 = timeSub(
  ['[HEAD_TIME]', '[ARG_HOUR_12]', '[ARG_MIN_00]'],
  ['[HEAD_DURATION]', '[ARG_HOUR_02]', '[ARG_MIN_00]']
);
assert(sub3.resultString === '10:00', 'Multi-hour sub: 12:00 - 2h = 10:00');

// ---------------------------------------------------------------------------
// ROUTE_DURATION_BETWEEN
// ---------------------------------------------------------------------------
console.log('\n=== ROUTE_DURATION_BETWEEN ===');

// Simple: 14:20 to 15:00 = 40 minutes
const dur1 = durationBetween(
  ['[HEAD_TIME]', '[ARG_HOUR_14]', '[ARG_MIN_20]'],
  ['[HEAD_TIME]', '[ARG_HOUR_15]', '[ARG_MIN_00]']
);
assert(dur1.totalMinutes === 40, 'Duration: 14:20 to 15:00 = 40 minutes');

// Cross-day: 23:00 to 01:00 = 2 hours
const dur2 = durationBetween(
  ['[HEAD_TIME]', '[ARG_HOUR_23]', '[ARG_MIN_00]'],
  ['[HEAD_TIME]', '[ARG_HOUR_01]', '[ARG_MIN_00]']
);
assert(dur2.totalMinutes === 120, 'Cross-day: 23:00 to 01:00 = 120 minutes');

// ROUTE_CALENDAR_SHIFT and ROUTE_TIMEZONE_CONVERT are v2 scope
console.log('\n=== V2 Scope Operations ===');

let calThrown = false;
try {
  execute(parseRoutingCall('[ROUTE_CALENDAR_SHIFT] [HEAD_TIME] [ARG_HOUR_09] [ARG_MIN_00]'));
} catch (e) {
  calThrown = e.message.includes('not implemented in v1');
}
assert(calThrown, 'CALENDAR_SHIFT throws v2 not-implemented error');

let tzThrown = false;
try {
  execute(parseRoutingCall('[ROUTE_TIMEZONE_CONVERT] [HEAD_TIME] [ARG_HOUR_14] [ARG_MIN_00]'));
} catch (e) {
  tzThrown = e.message.includes('not implemented in v1');
}
assert(tzThrown, 'TIMEZONE_CONVERT throws v2 not-implemented error');


// ---------------------------------------------------------------------------
// execute() Dispatcher
// ---------------------------------------------------------------------------
console.log('\n=== execute() Dispatcher ===');

// Via dispatcher: 09:58 + 5m
const exec1 = execute(parseRoutingCall(
  '[ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_09] [ARG_MIN_58] [HEAD_DURATION] [ARG_MIN_05]'
));
assert(exec1.resultString === '10:03', 'execute(TIME_ADD): 09:58 + 5m = 10:03');

// Via dispatcher: subtraction
const exec2 = execute(parseRoutingCall(
  '[ROUTE_TIME_SUB] [HEAD_TIME] [ARG_HOUR_00] [ARG_MIN_05] [HEAD_DURATION] [ARG_MIN_10]'
));
assert(exec2.resultString === '23:55', 'execute(TIME_SUB): 00:05 - 10m = 23:55');

// Via dispatcher: duration between with REF resolution
const exec3 = execute(
  parseRoutingCall('[ROUTE_DURATION_BETWEEN] [REF_1] [REF_2]'),
  {
    references: {
      '[REF_1]': ['[HEAD_TIME]', '[ARG_HOUR_14]', '[ARG_MIN_20]'],
      '[REF_2]': ['[HEAD_TIME]', '[ARG_HOUR_15]', '[ARG_MIN_00]'],
    }
  }
);
assert(exec3.totalMinutes === 40, 'execute(DURATION_BETWEEN) with REF resolution: 40 minutes');

// Via dispatcher: TIME_ADD with REF resolution (Issue #7 fix)
const exec4 = execute(
  parseRoutingCall('[ROUTE_TIME_ADD] [REF_1] [HEAD_DURATION] [ARG_MIN_10]'),
  {
    references: {
      '[REF_1]': ['[HEAD_TIME]', '[ARG_HOUR_09]', '[ARG_MIN_50]'],
    }
  }
);
assert(exec4.resultString === '10:00', 'execute(TIME_ADD) with REF resolution: 09:50 + 10m = 10:00');

// parseDurationTokens throws on garbage (Issue #15)
let durThrown = false;
try {
  parseDurationTokens(['[HEAD_DURATION]']);
} catch (e) {
  durThrown = true;
}
assert(durThrown, 'parseDurationTokens throws on empty duration');

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------
console.log(`\n${'='.repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed out of ${passed + failed}`);
console.log(`${'='.repeat(50)}`);

process.exit(failed > 0 ? 1 : 0);
