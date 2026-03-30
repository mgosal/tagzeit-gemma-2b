#!/usr/bin/env node
/**
 * CLI wrapper around temporal_engine.js for Python interop.
 * 
 * Usage:
 *   node compute_route.js '[ROUTE_TIME_ADD] [HEAD_TIME] [ARG_HOUR_09] [ARG_MIN_58] [HEAD_DURATION] [ARG_MIN_05]'
 *   → {"resultString":"10:03","resultTokens":["[HEAD_TIME]","[ARG_HOUR_10]","[ARG_MIN_03]"]}
 * 
 * Or pipe via stdin:
 *   echo '[ROUTE_TIME_ADD] ...' | node compute_route.js --stdin
 */

const { parseRoutingCall, execute } = require('./temporal_engine.js');

function computeRoute(routeString) {
  try {
    const parsed = parseRoutingCall(routeString);
    if (!parsed) {
      return { error: 'NO_ROUTE_TOKENS', input: routeString };
    }
    const result = execute(parsed);
    return {
      operation: parsed.operation,
      resultString: result.resultString,
      resultTokens: result.resultTokens || null,
      dayOverflow: result.dayOverflow || result.dayUnderflow || false,
    };
  } catch (err) {
    return { error: err.message, input: routeString };
  }
}

// --- Main ---
if (process.argv.includes('--stdin')) {
  // Read from stdin (for batch processing)
  let data = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => data += chunk);
  process.stdin.on('end', () => {
    const lines = data.trim().split('\n');
    const results = lines.map(line => computeRoute(line.trim()));
    console.log(JSON.stringify(results));
  });
} else if (process.argv.length > 2) {
  // Single route string as argument
  const routeString = process.argv.slice(2).join(' ');
  const result = computeRoute(routeString);
  console.log(JSON.stringify(result));
} else {
  console.error('Usage: node compute_route.js "<route tokens>"');
  console.error('   or: echo "<route tokens>" | node compute_route.js --stdin');
  process.exit(1);
}
