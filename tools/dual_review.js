export const meta = {
  name: 'pymars-dual-review',
  description: 'Two independent reviews (spec and adversarial) of one pymars pull request',
  phases: [{ title: 'Review', detail: 'two reviewers check the same head independently and each posts a verdict' }],
}

const B = '/Users/aschuler/Documents/research/projects/pymars/.worktrees/journal/briefs'
const VERDICT = {
  type: 'object',
  properties: {
    role: { type: 'string' },
    verdict: { type: 'string', enum: ['APPROVE', 'REQUEST_CHANGES'] },
    head_sha: { type: 'string' },
    gate_b: { type: 'string', description: 'PASS or FAIL, and the gate B log path' },
    comment_url: { type: 'string' },
    blocking: { type: 'array', items: { type: 'string' }, description: 'one line per blocking finding' },
    nonblocking: { type: 'array', items: { type: 'string' }, description: 'one line per non-blocking finding' },
  },
  required: ['role', 'verdict', 'head_sha', 'blocking'],
}
const ROLE = {
  spec: 'Your role is `spec`: check the change against docs/algorithm.md, VALIDATION_PLAN.md and the sources the spec cites. For a spec change, check each rule against its cited source, rerun the black-box scripts it cites, and judge whether an implementer who reads only the spec could write the code without guessing.',
  adversarial: 'Your role is `adversarial`: try to break the change. Look for inputs, edge cases, orderings and readings that give a wrong or ambiguous result. For a spec change, look for rules that are ambiguous, contradictory, incomplete or untestable, and test the rules against fresh black-box earth runs of your own on new datasets (follow the clean room strictly). You may start one helper to search for counterexamples. Report each failure with a minimal reproduction.',
  single: 'Your role is `single`: the one reviewer; cover the spec view and the adversarial view in proportion to the risk.',
}
const a = args

function prompt(role) {
  return [
    `You review pull request #${a.pr} in alejandroschuler/mars: ${a.title}. You did not write it.`,
    ROLE[role],
    `Read and follow ${B}/COMMON.md and ${B}/REVIEWER.md, and use \`${role}\` as the role in the verdict line. The author's brief: ${a.brief}.`,
    `The expected head SHA is ${a.head}. If the pull request head differs, review the actual head and say so.`,
    a.focus ? `Points that need care:\n${a.focus}` : '',
    'Post exactly one verdict comment as REVIEWER.md says. Do not push, merge or change labels. Remove your review worktree when done. Your final answer is the structured result.',
  ].filter(Boolean).join('\n\n')
}

phase('Review')
const results = await parallel(a.roles.map(role => () =>
  agent(prompt(role), { label: `review:${role}`, phase: 'Review', schema: VERDICT })))
return results