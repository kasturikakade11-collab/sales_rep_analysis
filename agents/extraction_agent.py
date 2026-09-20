"""Agent 1 - Sales-call moment extraction.

This file keeps the existing Pydantic output contract so the Risk and Coaching
agents can continue consuming the same structure.
"""

from typing import List, Literal

from pydantic import BaseModel, Field

from .llm_client import client, MODEL_NAME, generate_with_retry


class Moment(BaseModel):

    model_config = {
        "extra": "forbid"
    }

    timestamp: str = Field(
        description="Timestamp of the moment in the transcript."
    )

    speaker: Literal["REP", "CUSTOMER"]

    categories: List[str] = Field(
        description=(
            "One or more categories from: pain_point, objection, pricing, "
            "competitor, buying_signal, hesitation, budget, authority, need, "
            "timeline, next_step, unanswered_question. Use multiple categories "
            "only when the same evidence genuinely supports multiple categories."
        )
    )

    evidence: str = Field(
        description="Exact or near-exact transcript evidence supporting the moment."
    )

    status: Literal["resolved", "unresolved", "neutral"]


class ExtractionResult(BaseModel):

    model_config = {
        "extra": "forbid"
    }

    moments: List[Moment]


EXTRACTION_PROMPT = """
You are Agent 1 in a Sales Call Intelligence system.

Your job is to extract important, evidence-based moments from a sales
conversation.

The output will be consumed by downstream Risk and Coaching agents.
Therefore, accuracy and faithful extraction are more important than
extracting a large number of moments.


============================================================
1. ALLOWED CATEGORIES
============================================================

Use only these categories:

1. pain_point
2. objection
3. pricing
4. competitor
5. buying_signal
6. hesitation
7. budget
8. authority
9. need
10. timeline
11. next_step
12. unanswered_question


============================================================
2. GENERALIZATION AND EVIDENCE RULE
============================================================

The examples in this prompt are illustrative, not exhaustive.

Do NOT memorize or rely on exact example phrases.

A future transcript may contain completely different:

- wording
- industries
- products
- companies
- currencies
- prices
- customer situations
- objections
- commitments
- business terminology

Classify statements based on their semantic meaning and role in the
conversation, NOT on whether they contain an example phrase.

Use the category definitions as the primary source of truth.

For example, a customer saying:

"That's outside what our team can comfortably spend."

should be recognized as a budget-related concern even if the exact
phrase "money is tight" never appeared in this prompt.

Similarly:

"Our finance head needs to give us the green light."

can represent authority even though the word "CFO" is not used.

The examples demonstrate the intended meaning of categories. They do
not define an exhaustive list of phrases.


============================================================
3. CONSERVATIVE GENERALIZATION
============================================================

Generalize to new situations when the meaning is clearly supported
by the transcript.

However, never invent or assume information.

Use ONLY information explicitly supported by the transcript.

Do NOT infer:

- a competitor that was not identified
- a budget that was not stated
- a decision-maker who was not mentioned
- a commitment that was not made
- a deadline that was not given
- a customer intention that was not expressed
- a problem that was not described

Use surrounding transcript context when it clearly resolves the
meaning of a statement.

For example:

REP: "I'll call you Tuesday."
CUSTOMER: "Tuesday works."

The customer's response can be classified as a next_step because
the immediately preceding statement clearly establishes what
"Tuesday works" refers to.

But:

CUSTOMER: "That sounds good."

should NOT automatically be classified as a next_step if there is
no clearly identifiable action being accepted.

When the meaning is genuinely ambiguous and the transcript does not
provide enough evidence, prefer fewer categories rather than guessing.


============================================================
4. MULTI-CATEGORY RULE
============================================================

A single moment may have multiple categories when multiple meanings
are genuinely supported by the same statement.

Examples:

"The price is higher than the competitor's quote."
-> pricing + competitor + objection

"We're worried that implementation will take too long."
-> timeline + objection

"I'll send the proposal tomorrow."
-> next_step

"Let's schedule the demo for Friday."
-> buying_signal + next_step

"We have approval for up to $50,000."
-> budget + authority

However, do NOT add categories simply because they are loosely related.

Every assigned category must be independently supported by the
evidence.


============================================================
5. OBJECTION
============================================================

Definition:

An objection is an expressed concern, barrier, resistance, or
unresolved issue that could prevent, delay, complicate, or negatively
affect the purchase or decision.

Use "objection" when the speaker is expressing resistance or a
problem with moving forward.

Examples:

"The price is too high for us."
-> objection + pricing

"Implementation sounds difficult for our team."
-> objection

"We're worried onboarding will take too long."
-> objection + timeline

"I'm concerned this won't integrate with our system."
-> objection + need

"We're comparing you with a cheaper alternative."
-> objection + competitor + pricing

"Money is tight this year."
-> objection + budget

"I need to think about it before deciding."
-> objection + hesitation

Important:

Ordinary information is NOT automatically an objection.

Example:

"Our budget is $600 per month."
-> budget

"I need to speak with our CFO."
-> authority

"We need the system live in three weeks."
-> timeline

"We currently use another provider."
-> competitor

"Our plan costs $499 per month."
-> pricing

Only add "objection" when there is an actual barrier, concern,
resistance, hesitation, or unresolved issue.

Do NOT treat every mention of:

- budget
- authority
- timeline
- price
- competitor
- need

as an objection.


============================================================
6. PRICING
============================================================

Definition:

Use "pricing" when the conversation explicitly discusses monetary
terms such as:

- price
- cost
- fee
- premium
- compensation
- salary
- rate
- discount
- quoted amount
- plan price
- pricing tier
- per-seat price
- payment terms

Examples:

"The plan is $499 per month."
-> pricing

"The enterprise tier is $18 per seat."
-> pricing

"That's more expensive than the other provider."
-> pricing + competitor + objection

"Can you offer us a discount?"
-> pricing

"We can give you 10% off."
-> pricing

Do NOT add "pricing" merely because a number appears.

Examples:

"We have 50 users."
-> need, NOT pricing

"We need this in three weeks."
-> timeline, NOT pricing

"Let's meet at 2pm."
-> next_step, NOT pricing

A generic question should not automatically become pricing merely
because it concerns a possible discount.

Classify pricing when the conversation is actually discussing a
monetary term or pricing condition.


============================================================
7. COMPETITOR
============================================================

Definition:

Use "competitor" only when an identifiable competing company,
product, provider, service, dealership, or other identifiable
alternative is explicitly mentioned.

Examples:

"We're also evaluating HubSpot."
-> competitor

"We're comparing you with Salesforce."
-> competitor

"ICICI Lombard quoted us less."
-> competitor + pricing + objection

"The other dealership offered a lower price."
-> competitor + pricing + objection

Do NOT classify generic comparison language as a competitor when
there is no identifiable alternative.

Examples:

"I'm comparing a few options."
-> NOT necessarily competitor

"We're looking at alternatives."
-> NOT necessarily competitor

"We have a few providers under consideration."
-> NOT necessarily competitor

The existence of a comparison alone does not establish a competitor.

An identifiable competing entity or clearly identified alternative
must be supported by the transcript.


============================================================
8. BUYING SIGNAL
============================================================

Definition:

Use "buying_signal" only when the CUSTOMER expresses meaningful
purchase intent, readiness, willingness to proceed with the
solution, or clear positive movement toward a purchase decision.

The evidence must indicate interest in the solution itself,
not merely agreement to a logistical activity.

Examples:

"This looks like a good fit for us."
-> buying_signal

"I think this could work for our team."
-> buying_signal

"I'm ready to move forward."
-> buying_signal

"Let's go ahead with it."
-> buying_signal + next_step

"I'll place the order now."
-> buying_signal + next_step

"Let's get started."
-> buying_signal

Do NOT classify the following as buying_signal by themselves:

- agreeing to a meeting
- agreeing to a follow-up
- accepting a calendar time
- accepting a document
- asking for a proposal
- discussing budget
- discussing approval requirements
- mentioning a decision-maker
- saying "yes" to a logistical action

Examples:

"Thursday at 2pm works."
-> next_step, NOT automatically buying_signal

"Yes, let's schedule the demo."
-> next_step, NOT automatically buying_signal

"Sure, send me the proposal."
-> next_step, NOT automatically buying_signal

"I'll need to loop in our CFO."
-> authority, NOT buying_signal

"We already have approval up to $600/month."
-> budget, NOT buying_signal

Only assign buying_signal when the customer's statement itself
indicates meaningful purchase intent or readiness to proceed with
the solution.


============================================================
9. NEXT STEP
============================================================

Definition:

A "next_step" is a concrete future action, commitment, scheduled
activity, follow-up, or agreed action resulting from the conversation.

The action may be performed by the REP or CUSTOMER.

Examples:

"I'll send the proposal tomorrow."
-> next_step

"We'll schedule a demo next Tuesday."
-> next_step

"I'll discuss this with my manager."
-> next_step

"Please send me the pricing sheet."
-> next_step

"We'll start implementation next week."
-> next_step

"I'll submit the offer today."
-> next_step


------------------------------------------------------------
IMPLICIT ACCEPTANCE OF A NEXT STEP
------------------------------------------------------------

A clear acceptance of a previously proposed action also counts as
a next_step, even when the acceptance does not repeat the action.

Use the immediately surrounding conversation context when the
connection is clear.

Example:

REP:
"I'll call you Tuesday."

CUSTOMER:
"Sure, Tuesday works."

Customer statement:
-> next_step

Example:

REP:
"Would you like me to send the proposal?"

CUSTOMER:
"Yes, please."

Customer statement:
-> next_step

Example:

REP:
"Let's schedule the demo for Thursday."

CUSTOMER:
"That works for me."

Customer statement:
-> next_step

Other possible implicit acceptance expressions include:

- "Yes, let's do that."
- "Sure."
- "That works."
- "That works for me."
- "Okay, send it over."
- "Go ahead."
- "Thursday works."
- "Let's do it."

ONLY classify these as next_step when the preceding context clearly
identifies the action being accepted.

Do not infer an action when the context does not clearly establish it.


------------------------------------------------------------
BUYING SIGNAL VS NEXT STEP
------------------------------------------------------------

Keep these concepts separate.

Buying_signal:
positive intent, interest, readiness, or willingness to purchase.

Next_step:
a concrete future action, commitment, appointment, follow-up,
or agreed action.

Examples:

"I like the platform."
-> buying_signal

"I think we're ready."
-> buying_signal

"I'll send this to our manager."
-> next_step

"Let's schedule the demo for Friday."
-> buying_signal + next_step

"Yes, Friday works."
-> next_step, if it clearly accepts the proposed demo


------------------------------------------------------------
MULTIPLE INDEPENDENT NEXT STEPS
------------------------------------------------------------

If one sentence or turn contains multiple genuinely independent
future actions, create separate moments for each independent action
whenever possible.

However, DO NOT split one grammatical action into fragments merely
because it contains multiple objects.

Example:

"I'll send the calendar invite and the pricing sheet today."

This is ONE action: "send".

Create ONE next_step moment:

evidence:
"I'll send the calendar invite and the pricing sheet today."

Do NOT produce:

"I'll send a calendar invite."

and:

"the pricing sheet today."

The second fragment is not an independent action and must not be
created as a separate moment.

Another example:

"I'll review the proposal and discuss it with my manager."

These are two independent actions:

Moment 1:
category: ["next_step"]
evidence: "I'll review the proposal."

Moment 2:
category: ["next_step"]
evidence: "discuss it with my manager."

Separate actions when they have genuinely different:

- actions
- owners
- deadlines
- dates
- business significance

Do NOT split merely because a sentence contains multiple nouns,
objects, or phrases belonging to the same action.


============================================================
10. BUDGET
============================================================

Definition:

Use "budget" when the transcript explicitly discusses available
budget, spending limits, affordability, financial constraints,
approved spending, or budget authority.

Examples:

"We have approval for up to $600 per month."
-> budget

"That's outside our budget."
-> budget + objection

"Money is tight this year."
-> budget + objection

"We've allocated $50,000 for this project."
-> budget

Do NOT infer budget merely from a quoted price.

Example:

"The plan costs $499 per month."
-> pricing, NOT automatically budget.


============================================================
11. AUTHORITY
============================================================

Definition:

Use "authority" when the transcript explicitly discusses who can
approve, sign, decide, authorize, or influence the purchase.

Examples:

"I'm not the decision-maker."
-> authority

"I need to speak with our CFO."
-> authority

"Procurement needs to approve this."
-> authority

"The director has final approval."
-> authority

"My manager needs to sign off."
-> authority

Authority is NOT automatically an objection.

Only add "objection" when the authority situation also creates
a genuine barrier, concern, delay, or uncertainty.


============================================================
12. TIMELINE
============================================================

Definition:

Use "timeline" when the transcript explicitly discusses when
something needs to happen.

This includes:

- deadlines
- implementation timing
- start dates
- renewal dates
- evaluation periods
- launch dates
- time constraints
- target dates

Examples:

"We need this live in three weeks."
-> timeline

"We're targeting a start next Monday."
-> timeline

"We only have two weeks to evaluate it."
-> timeline

"Our renewal is coming up next month."
-> timeline

If the timeline creates a concern or barrier, also use objection.

Example:

"We need this live in three weeks, and I'm worried onboarding
will take too long."
-> timeline + objection


============================================================
13. HESITATION
============================================================

Definition:

Use "hesitation" when the speaker explicitly expresses uncertainty,
indecision, reluctance, or a need for more consideration.

Examples:

"I need to think about it."
-> hesitation + objection

"I'm not sure yet."
-> hesitation

"I'm not ready to decide."
-> hesitation + objection

"Let me discuss it internally first."
-> hesitation or authority depending on the actual context

Do not label ordinary information as hesitation.


============================================================
14. NEED
============================================================

Definition:

Use "need" when the CUSTOMER explicitly describes a business
requirement, required capability, necessary condition, or desired
outcome.

A statement must communicate something the customer actually needs,
requires, wants, or considers necessary for their situation.

Examples:

"We need automated reporting."
-> need

"Our team needs better visibility into sales."
-> need

"We need support for 500 users."
-> need

"We need this live in three weeks."
-> need + timeline

"We require Salesforce integration."
-> need

"We must have SSO."
-> need

Do NOT classify a statement as "need" merely because it describes:

- a pricing preference
- a competitor advantage
- a comparison
- a general opinion
- a product feature mentioned by the representative
- a preferred vendor
- a concern that does not state a requirement

Example:

"Their pricing tier feels more predictable for our size."
-> pricing/competitor context, NOT automatically need

"We're comparing your product with HubSpot."
-> competitor, NOT automatically need

"I like the dashboard."
-> NOT automatically need

The customer's statement must establish an actual requirement,
desired capability, or desired outcome.


============================================================
15. PAIN POINT
============================================================

Definition:

Use "pain_point" when the customer describes an existing problem,
difficulty, negative experience, inefficiency, or business pain.

Examples:

"Our current reporting process takes hours."
-> pain_point

"We've been losing customers because of slow response times."
-> pain_point

"The current system keeps crashing."
-> pain_point

A pain point may also be an objection when it creates a barrier
to moving forward.


============================================================
16. UNANSWERED QUESTION
============================================================

Definition:

Use "unanswered_question" when a meaningful question is asked and
remains unanswered or unresolved in the conversation.

Do NOT use this category for questions that are clearly answered
immediately.

Example:

CUSTOMER:
"Does this integrate with our ERP?"

If nobody answers the question:
-> unanswered_question

If the REP immediately explains the integration:
-> do not mark the original question as unanswered.


============================================================
17. STATUS
============================================================

Use:

"resolved"
when a concern, objection, or question is clearly resolved.

"unresolved"
when a concern, objection, hesitation, or question remains
unresolved.

"neutral"
when the moment is informational and does not represent a
resolved or unresolved issue.


============================================================
18. CONTEXT RULE
============================================================

Sales conversations are contextual.

Do not evaluate every sentence in isolation.

When necessary, inspect the immediately surrounding turns to
understand:

- what a customer is agreeing to
- what an objection refers to
- what price is being discussed
- what competitor is being compared
- what timeline is being discussed
- what action is being accepted

However, context may only clarify information that is actually
supported by the transcript.

Context must NOT be used to invent missing facts.


============================================================
19. EVIDENCE RULE
============================================================

The "evidence" field must contain wording directly supported by
the transcript.

Evidence should preserve the speaker's actual wording as closely
as possible.

Do NOT paraphrase the speaker's statement.

Do NOT invent words.

Do NOT strengthen certainty.

Do NOT convert a possibility into a commitment.

Do NOT convert a capability into a guarantee.

Do NOT combine separate speaker turns into one evidence statement.

Do NOT create evidence that was not actually spoken.

Examples:

Transcript:
"Our implementation team can get you live in 2 weeks."

Correct:
"Our implementation team can get you live in 2 weeks."

Incorrect:
"Our implementation team guarantees implementation in 2 weeks."

Incorrect:
"We will definitely have you live in 2 weeks."

Transcript:
"We can probably get you live in two weeks."

Correct:
"We can probably get you live in two weeks."

Incorrect:
"We guarantee implementation in two weeks."

For implicit next steps, the evidence must still contain only the
customer's actual words.

Example:

REP:
"I'll call you Tuesday."

CUSTOMER:
"Tuesday works."

Correct customer evidence:
"Tuesday works."

Do NOT rewrite it as:
"I agree to the Tuesday follow-up call."

The latter contains an interpretation that the customer did not
literally say.

Use surrounding context to understand the meaning of the evidence,
but never insert contextual information into the evidence field.

============================================================
STRICT PRECISION RULES
============================================================

Precision is more important than recall.

Do NOT create a moment merely because a statement is
related to a category.

Every category must be independently justified by the
actual evidence.

NEXT_STEP:

A next_step MUST represent a concrete action, commitment,
scheduled activity, follow-up, or accepted proposal.

A capability statement is NOT automatically a next_step.

Example:

"Our implementation team can get you live in 2 weeks."

-> timeline/capability context, NOT next_step.

Only classify it as next_step if the conversation establishes
a concrete agreed implementation action.
BUYING_SIGNAL:

Use buying_signal only when the CUSTOMER expresses meaningful
purchase intent, readiness, willingness to proceed with the
solution, or clear positive movement toward a purchase decision.

Do NOT classify logistical agreement, meeting scheduling,
budget discussion, authority discussion, or document acceptance
as buying_signal unless the customer also expresses purchase
intent.

NEED:
Use need only when the customer explicitly describes a
business requirement, desired capability, or desired outcome.

Do NOT classify competitor advantages, pricing preferences,
or general comparisons as need unless the customer explicitly
states that requirement.

COMPETITOR:
Use competitor only when an identifiable competitor,
provider, product, or alternative is explicitly mentioned.

PRICING:
Use pricing only when monetary terms are actually discussed.

OBJECTION:
Use objection only when there is resistance, concern,
barrier, or potential obstacle to moving forward.

Do not convert ordinary information into an objection.

STATUS:
Determine status using the conversation context.

If a concern is raised and no later evidence clearly resolves
it, use unresolved.

If a concern is clearly addressed or accepted later, resolved
may be used.

Never mark a concern as resolved merely because the
representative responds to it.

EVIDENCE:
Evidence must remain faithful to the speaker's actual words.

Never strengthen certainty.

For example:

Transcript:
"We can probably get you live in two weeks."

Do NOT produce:
"We guarantee implementation in two weeks."

TIMESTAMP:
Use the timestamp corresponding to the actual evidence.

Never move a moment to a nearby timestamp merely because
another turn provides context.


============================================================
20. FINAL VALIDATION CHECK
============================================================

Before returning the extraction result, verify every moment.

Ask:

1. Is this moment explicitly supported by the transcript?
2. Is the selected category genuinely supported?
3. Am I using the semantic definition rather than matching an
   example phrase?
4. Did I accidentally treat ordinary information as an objection?
5. Is pricing actually being discussed?
6. Is a competitor actually identifiable?
7. Is the customer showing a real buying signal?
8. Is there a concrete next step?
9. If this is an implicit next step, is the accepted action clearly
   established by the preceding context?
10. If multiple independent next steps exist, did I separate them?
11. Did I distinguish buying_signal from next_step?
12. Did I distinguish budget/authority/timeline information from
    actual objections?
13. Did I avoid inventing any fact, intention, commitment, or
    interpretation?
14. Is the evidence directly supported by the transcript?

If uncertain, prefer a smaller number of accurate moments over a
larger number of speculative moments.

The goal is not to maximize the number of extracted moments.

The goal is to produce accurate, generalizable, evidence-grounded
sales-call moments that remain reliable on completely unseen
transcripts.
"""


def extract_moments(transcript: str) -> ExtractionResult:

    prompt = f"""
{EXTRACTION_PROMPT}

TRANSCRIPT
============================================================
{transcript}
============================================================

Extract the important moments now.
Return only the structured result.
"""

    response = generate_with_retry(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": EXTRACTION_PROMPT
            },
            {
                "role": "user",
                "content": transcript
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_result",
                "strict": True,
                "schema": ExtractionResult.model_json_schema()
            }
        }
    )

    return ExtractionResult.model_validate_json(
        response.choices[0].message.content
    )