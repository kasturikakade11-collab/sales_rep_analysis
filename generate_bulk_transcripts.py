import csv
import json
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth.json"
DATASET_INDEX_PATH = DATA_DIR / "dataset_index.csv"

TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# 8 Core Domains x 9 variations each = 72 new calls
DOMAINS = [
    {
        "domain": "b2b_sales",
        "prefix": "b2b_saas",
        "products": [
            "CloudTrack",
            "Zynq AI Invoicing",
            "DataDeck",
            "ChatForge",
        ],
        "competitors": ["Monday.com", "Zoho", "HubSpot", "Salesforce"],
        "objections": [
            "worried about learning curve for the team",
            "pricing seems high vs current tool",
            "needs approval from finance and leadership",
            "unsure if it integrates with existing stack",
        ],
        "outcomes": [
            "next call booked",
            "deal won on the spot",
            "customer needs to consult their team",
            "deal lost",
        ],
        "reps": ["Aisha", "Karan", "Priya", "Rahul"],
        "prospects": ["Rohan", "Sneha", "Arjun", "Neha"],
    },
    {
        "domain": "call_center",
        "prefix": "support_telecom",
        "products": ["SwiftFiber Broadband", "ConnectPlus Mobile", "SkyView DTH"],
        "competitors": ["Airtel", "Jio", "Tata Play"],
        "objections": [
            "unexpected overage charges on bill",
            "recurrent intermittent packet drops",
            "app crashing during payment gateway sync",
        ],
        "outcomes": [
            "issue resolved + upsell accepted",
            "issue resolved + upsell declined",
            "issue unresolved/escalated",
        ],
        "reps": ["Priya", "Sana", "Meera", "Rahul"],
        "prospects": ["Deepak", "Anita", "Suresh", "Farhan"],
    },
    {
        "domain": "ecommerce_chat",
        "prefix": "ecommerce_retail",
        "products": [
            "StreetGlide Sneakers",
            "GlowLab Peptide Serum",
            "ErgoDesk Pro",
        ],
        "competitors": ["Nike", "Ordinary", "IKEA"],
        "objections": [
            "uncertainty about true-to-size fit",
            "coupon code expired at checkout",
            "worried about return shipping costs",
        ],
        "outcomes": [
            "purchase completed",
            "discount given and purchase made",
            "customer asks to speak to human",
        ],
        "reps": ["Nisha", "Aman", "Riya", "Zara"],
        "prospects": ["Vikram", "Tanya", "Rohit", "Simran"],
    },
    {
        "domain": "automotive_sales",
        "prefix": "auto_motors",
        "products": [
            "Apex Turbo SUV",
            "VoltEV Hatchback",
            "CityGlide Sedan",
        ],
        "competitors": ["Hyundai", "Tata", "Maruti", "Honda"],
        "objections": [
            "price negotiation against competitor quotation",
            "demands higher trade-in valuation for old vehicle",
            "range anxiety and fast-charging availability",
        ],
        "outcomes": [
            "deal closed",
            "test drive booked",
            "financing discussion scheduled",
            "customer walks away to compare",
        ],
        "reps": ["Gaurav", "Rohit", "Deepa", "Nisha"],
        "prospects": ["Amit", "Sunita", "Karan", "Preeti"],
    },
    {
        "domain": "real_estate",
        "prefix": "realty_properties",
        "products": [
            "2BHK Luxury Apartment",
            "High Street Commercial Shop",
            "Gated Community Villa Plot",
        ],
        "competitors": ["MagicBricks", "99acres", "Direct Builder Sales"],
        "objections": [
            "price is above current budget ceiling",
            "worried about location connectivity and road width",
            "needs to inspect property with family elder",
        ],
        "outcomes": [
            "site visit scheduled",
            "offer made",
            "client needs time to decide",
        ],
        "reps": ["Ramesh", "Shalini", "Neelam", "Ajay"],
        "prospects": ["Harish", "Pooja", "Sanjay", "Meenal"],
    },
    {
        "domain": "insurance",
        "prefix": "insurance_policy",
        "products": [
            "CareGuard Health Floater",
            "Apex Shield Term Life",
            "DriveSafe Bumper-to-Bumper",
        ],
        "competitors": ["Star Health", "LIC", "ICICI Lombard", "HDFC Life"],
        "objections": [
            "worried about hospital cashless claim hassles",
            "confused about room rent capping deductions",
            "hesitant on annual premium commitment",
        ],
        "outcomes": [
            "policy purchased",
            "follow-up call scheduled",
            "customer needs family consultation",
        ],
        "reps": ["Suman", "Radhika", "Vikas", "Manoj"],
        "prospects": ["Anil", "Rajesh", "Swati", "Pallavi"],
    },
    {
        "domain": "customer_success",
        "prefix": "am_renewal",
        "products": [
            "MetricPulse Analytics",
            "KubeShield Cloud Defense",
            "TaskFlow Enterprise",
        ],
        "competitors": ["CloudSentry", "PowerBI", "Asana"],
        "objections": [
            "inherited account after champion left company",
            "mandatory departmental software budget freeze",
            "unhappy with a recent patch regression outage",
        ],
        "outcomes": [
            "renewed at same tier",
            "renewed + upsell accepted",
            "needs more time to decide",
        ],
        "reps": ["Alok", "Ritika", "Sameer", "Divya"],
        "prospects": ["Sonal", "Vivek", "Nikhil", "Anjali"],
    },
    {
        "domain": "recruitment",
        "prefix": "talent_hiring",
        "products": [
            "Staffing Pipeline",
            "Tech Headhunting",
            "Executive Search",
        ],
        "competitors": ["Direct Inhouse Recruiting", "External Agency"],
        "objections": [
            "candidate holding a competing written offer",
            "hesitation about short-notice city relocation",
            "base compensation lower than market hike expectation",
        ],
        "outcomes": [
            "counter-offer discussion scheduled",
            "candidate accepts",
            "candidate needs more time",
        ],
        "reps": ["Anjali", "Kavita", "Rohit", "Suresh"],
        "prospects": ["Manoj", "Priyanka", "Ashish", "Neha"],
    },
]


def generate_single_dialogue(rep, prospect, product, comp, obj, outcome, idx):
  """Generates a structured, natural conversation with increasing timestamps."""
  sec = 0

  def get_ts(s):
    m = s // 60
    r = s % 60
    return f"[{m:02d}:{r:02d}]"

  lines = []

  # Opening
  lines.append(
      f"{get_ts(sec)} Rep: Hi {prospect}, {rep} here regarding your inquiry on"
      f" {product}. How are you doing today?"
  )
  sec += random.randint(4, 7)
  lines.append(
      f"{get_ts(sec)} Customer: Hi {rep}. Doing alright, thanks for checking"
      " in. Just had a few questions about the details."
  )
  sec += random.randint(5, 8)

  # Exploration
  lines.append(
      f"{get_ts(sec)} Rep: Happy to help clear those up. What specific parts of"
      " the setup were you looking at?"
  )
  sec += random.randint(4, 7)
  lines.append(
      f"{get_ts(sec)} Customer: Well, honestly, my primary hesitation is that we"
      f" are {obj}."
  )
  sec += random.randint(5, 9)

  # Competitive Context & Objection
  if random.choice([True, False]):
    lines.append(
        f"{get_ts(sec)} Rep: Completely understand that concern. Are you"
        f" currently comparing our package with {comp}?"
    )
    sec += random.randint(5, 8)
    lines.append(
        f"{get_ts(sec)} Customer: Yes, exactly. We had evaluated {comp}"
        " earlier, so we're trying to figure out which approach makes the most"
        " sense."
    )
    sec += random.randint(6, 10)
    has_comp = True
  else:
    lines.append(
        f"{get_ts(sec)} Rep: That's an entirely valid hesitation. Many of our"
        " clients bring up that exact concern during their initial review."
    )
    sec += random.randint(5, 8)
    lines.append(
        f"{get_ts(sec)} Customer: Right, so we just want to ensure we don't"
        " run into friction down the road."
    )
    sec += random.randint(4, 8)
    has_comp = False

  # Value delivery & pricing discussion
  price_val = f"{random.randint(15, 85)},000 rupees"
  lines.append(
      f"{get_ts(sec)} Rep: To ensure smooth adoption, our current tier includes"
      f" guided onboarding and direct SLA support, locked in around {price_val}."
  )
  sec += random.randint(6, 10)
  lines.append(
      f"{get_ts(sec)} Customer: Okay, {price_val} is within range, but how"
      " does your team manage turnaround times when issues arise?"
  )
  sec += random.randint(6, 9)

  # Solution demonstration
  lines.append(
      f"{get_ts(sec)} Rep: We maintain a guaranteed 45-minute response window"
      " with an assigned technical manager, so your team is never waiting on"
      " generic ticket queues."
  )
  sec += random.randint(7, 11)
  lines.append(
      f"{get_ts(sec)} Customer: That definitely removes a lot of the risk we"
      " were worried about."
  )
  sec += random.randint(4, 7)

  # Resolution matching outcome
  if "booked" in outcome or "scheduled" in outcome or "visit" in outcome:
    lines.append(
        f"{get_ts(sec)} Rep: How about we set up a dedicated follow-up session"
        " next Tuesday at 11 AM to review the full implementation plan?"
    )
    sec += random.randint(6, 9)
    lines.append(
        f"{get_ts(sec)} Customer: That works well for me. Send the calendar"
        " invitation across."
    )
    sec += random.randint(4, 6)
    lines.append(
        f"{get_ts(sec)} Rep: Invitation is on its way, {prospect}. Looking"
        " forward to speaking then!"
    )
    next_step = "calendar invite sent for follow-up session next Tuesday at 11 AM"
    bant_gap = {
        "present": False,
        "reason": "timeline and next action clearly agreed upon",
    }

  elif (
      "won" in outcome
      or "accepted" in outcome
      or "closed" in outcome
      or "completed" in outcome
      or "purchased" in outcome
      or "offer" in outcome
  ):
    lines.append(
        f"{get_ts(sec)} Rep: If you're comfortable with the terms, I can share"
        " the confirmation link and activate your tier today."
    )
    sec += random.randint(5, 8)
    lines.append(
        f"{get_ts(sec)} Customer: Yes, let's proceed and lock this in right"
        " now."
    )
    sec += random.randint(4, 6)
    lines.append(
        f"{get_ts(sec)} Rep: Wonderful! Generating the secure agreement now."
        " Welcome aboard, {prospect}!"
    )
    next_step = "activation link sent, customer agreed to immediate signup"
    bant_gap = {
        "present": False,
        "reason": "customer committed to transaction with clear budget and need",
    }

  elif "lost" in outcome or "declined" in outcome or "walks" in outcome:
    lines.append(
        f"{get_ts(sec)} Rep: Is there any specific modification to the tier"
        " that would make this viable for your current cycle?"
    )
    sec += random.randint(5, 8)
    lines.append(
        f"{get_ts(sec)} Customer: At this point we've decided to hold off and"
        " stick with our existing setup. We will pass for now."
    )
    sec += random.randint(5, 7)
    lines.append(
        f"{get_ts(sec)} Rep: Completely respect that decision. Thank you for"
        f" your time today, {prospect}."
    )
    next_step = "none, prospect passed on offer"
    bant_gap = {
        "present": True,
        "reason": (
            "budget and timing misalignment resulted in deal not moving forward"
        ),
    }

  else:  # consult / time to decide / escalated
    lines.append(
        f"{get_ts(sec)} Rep: Take the time you need to review the proposal"
        " with your team."
    )
    sec += random.randint(4, 7)
    lines.append(
        f"{get_ts(sec)} Customer: Thanks, I'll go through the numbers"
        " internally and touch base when we are ready."
    )
    sec += random.randint(4, 6)
    lines.append(
        f"{get_ts(sec)} Rep: Sounds like a plan, {prospect}. Have a great rest"
        " of your day!"
    )
    next_step = "prospect to review proposal with internal stakeholders"
    bant_gap = {
        "present": True,
        "reason": "internal authority review required before final sign-off",
    }

  transcript_text = "\n".join(lines)
  ground_truth_entry = {
      "objections": [obj],
      "pricing_mentions": [price_val],
      "competitor_mentions": [comp] if has_comp else [],
      "next_steps": [next_step] if next_step != "none" else [],
      "bant_gap": bant_gap,
  }

  return transcript_text, ground_truth_entry


def build_bulk_transcripts():
  # 1. Load existing ground_truth.json
  gt_data = {}
  if GROUND_TRUTH_PATH.exists():
    try:
      with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    except Exception as e:
      print(f"Notice: Reading existing ground_truth: {e}")

  # 2. Check existing IDs in index to prevent duplicate rows
  existing_ids = set(gt_data.keys())
  if DATASET_INDEX_PATH.exists():
    try:
      with open(DATASET_INDEX_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
          existing_ids.add(row.get("id"))
    except Exception:
      pass

  new_csv_rows = []
  count = 0

  # 3. Generate 9 files per domain (8 domains * 9 = 72 files)
  for dom_cfg in DOMAINS:
    domain_name = dom_cfg["domain"]
    prefix = dom_cfg["prefix"]

    for i in range(1, 10):
      call_id = f"{prefix}_call_{i:02d}"

      # If already generated, skip or increment
      if call_id in existing_ids:
        call_id = f"{prefix}_batch2_{i:02d}"

      rep = random.choice(dom_cfg["reps"])
      prospect = random.choice(dom_cfg["prospects"])
      prod = random.choice(dom_cfg["products"])
      comp = random.choice(dom_cfg["competitors"])
      obj = random.choice(dom_cfg["objections"])
      outcome = random.choice(dom_cfg["outcomes"])

      transcript_txt, gt_entry = generate_single_dialogue(
          rep, prospect, prod, comp, obj, outcome, i
      )

      # Write pure .txt file
      out_path = TRANSCRIPTS_DIR / f"{call_id}.txt"
      out_path.write_text(transcript_txt, encoding="utf-8")

      # Store ground truth
      gt_data[call_id] = gt_entry

      # Record for CSV index
      new_csv_rows.append({
          "id": call_id,
          "domain": domain_name,
          "outcome": outcome,
          "file": f"data/transcripts/{call_id}.txt",
      })
      count += 1

  # 4. Save merged ground_truth.json
  with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
    json.dump(gt_data, f, indent=2, ensure_ascii=False)

  # 5. Append to dataset_index.csv
  csv_exists = DATASET_INDEX_PATH.exists()
  with open(DATASET_INDEX_PATH, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["id", "domain", "outcome", "file"], lineterminator="\n"
    )
    if not csv_exists:
      writer.writeheader()
    for r in new_csv_rows:
      writer.writerow(r)

  print("=" * 60)
  print(f"SUCCESS: Generated {count} new transcripts in data/transcripts/")
  print(f"Total entries in ground_truth.json: {len(gt_data)}")
  print(f"Appended {len(new_csv_rows)} records to {DATASET_INDEX_PATH}")
  print("=" * 60)


if __name__ == "__main__":
  build_bulk_transcripts()