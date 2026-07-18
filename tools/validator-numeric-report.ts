/**
 * Find every non-finite numeric leaf before a validator report can be signed
 * or serialized. JSON would silently coerce NaN and infinities to null, which
 * must never turn an undefined acceptance metric into a passing comparison.
 */
export function nonFiniteReportNumberPaths(
  value: unknown,
  rootPath = '$',
): readonly string[] {
  const paths: string[] = [];
  visit(value, rootPath, new Set<object>(), paths);
  return paths;
}

function visit(
  value: unknown,
  path: string,
  ancestors: Set<object>,
  paths: string[],
): void {
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) paths.push(path);
    return;
  }
  if (value === null || typeof value !== 'object') return;
  if (ancestors.has(value)) {
    throw new Error(`Validator report contains a cyclic object at ${path}`);
  }
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      value.forEach((item, index) =>
        visit(item, `${path}[${index}]`, ancestors, paths));
      return;
    }
    for (const [name, item] of Object.entries(value)) {
      visit(item, `${path}.${name}`, ancestors, paths);
    }
  } finally {
    ancestors.delete(value);
  }
}
