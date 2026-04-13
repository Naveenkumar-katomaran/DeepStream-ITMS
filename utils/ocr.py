import re
import logging as log
from collections import Counter

logging = log.getLogger(__name__)

# --- Indian Plate Standards ---
# Template: [S S] [N N] [L L L] [N N N N]
# Index:     0 1   2 3   4 5 6   7 8 9 10
EMPTY_TOKEN = "_"
STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JK', 'JH', 'KA', 'KL', 'LA', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OD', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK', 'UA', 'WB'
}

# Confusion character mappings (Letter slot vs numeric slot)
LETTER_FIX = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S', '6': 'G', '8': 'B'} 
NUMBER_FIX = {'O': '0', 'D': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'T': '7', 'C': '0'}

def normalize(text):
    if not text: return ""
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def correct_char(char, pos):
    """Context-aware character correction. Returns (char, was_corrected)."""
    if pos in [0, 1, 4, 5, 6]: # Alpha Slots
        c = LETTER_FIX.get(char, char) if char.isdigit() or char in ['O', 'D'] else char
        return c, (c != char)
    if pos in [2, 3, 7, 8, 9, 10]: # Numeric Slots
        c = NUMBER_FIX.get(char, char) if char.isalpha() else char
        return c, (c != char)
    return char, False

def align_to_template(text):
    """
    Tries multiple alignments (sliding window) to find the best fit for an 11-slot template.
    Returns: (best_aligned_list, best_score)
    """
    text = normalize(text)
    if len(text) < 4:
        return [EMPTY_TOKEN] * 11, -100

    best_aligned = [EMPTY_TOKEN] * 11
    max_score = -999

    # Try different sub-string lengths to handle leading/trailing noise
    for length in range(min(len(text), 12), 3, -1):
        for i in range(len(text) - length + 1):
            sub = text[i:i+length]
            
            # Sub-Alignment A: Front-Anchored (LL NN LLL NNNN)
            a_front = [EMPTY_TOKEN] * 11
            corr_a = 0
            for j in range(min(length, 11)):
                if j < 4: # State + RTO (0,1,2,3)
                    char, revised = correct_char(sub[j], j)
                    a_front[j] = char
                    if revised: corr_a += 1
                elif j < length - 4: # Middle Series (4,5,6)
                    a_idx = 4 + (j - 4)
                    if a_idx < 7: 
                        char, revised = correct_char(sub[j], a_idx)
                        a_front[a_idx] = char
                        if revised: corr_a += 1
                else: # Unique ID (7,8,9,10)
                    a_idx = 10 - (length - 1 - j)
                    if a_idx >= 7: 
                        char, revised = correct_char(sub[j], a_idx)
                        a_front[a_idx] = char
                        if revised: corr_a += 1
            
            # Sub-Alignment B: Back-Anchored (Always assumes the end is the Unique ID)
            a_back = [EMPTY_TOKEN] * 11
            corr_b = 0
            for j in range(min(length, 11)):
                t_idx = length - 1 - j
                a_idx = 10 - j
                if a_idx >= 0: 
                    char, revised = correct_char(sub[t_idx], a_idx)
                    a_back[a_idx] = char
                    if revised: corr_b += 1

            # Evaluate both scenarios and pick the one that makes more structural sense
            for aligned, corrections in [(a_front, corr_a), (a_back, corr_b)]:
                score = get_structure_score(aligned, corrections)
                # Extra Reward: If it's a full 9-11 char plate format match
                active_count = len([c for c in aligned if c != EMPTY_TOKEN])
                if active_count >= 9: score += 10 # Bonus for completeness
                
                if score > max_score:
                    max_score = score
                    best_aligned = aligned
                
    return best_aligned, max_score

def get_structure_score(aligned, corrections=0):
    """Calculates how closely a template alignment matches Indian plate rules."""
    score = 0
    # Mandatory State code (Slots 0,1)
    st = "".join([c for c in aligned[:2] if c != EMPTY_TOKEN])
    if st in STATE_CODES: 
        score += 40 # Huge boost for known state codes
    elif len(st) == 2 and st.isalpha(): 
        score += 5
    
    # District & Unique ID (Digits) - Mandatory format checks
    for i in [2, 3, 7, 8, 9, 10]:
        if aligned[i] != EMPTY_TOKEN:
            if aligned[i].isdigit(): score += 5
            else: score -= 15 # Severe penalty for letter in numeric slot

    # Series (Letters)
    for i in [4, 5, 6]:
        if aligned[i] != EMPTY_TOKEN:
            if aligned[i].isalpha(): score += 3
            else: score -= 10 # Increased penalty for number in series
            
    # Correction Penalty (Prefer authentic reads over auto-corrected ones)
    score -= (corrections * 3)
    
    # 10-Character Preference (The 95% standard in India: LL NN LL NNNN)
    active_chars = [c for c in aligned if c != EMPTY_TOKEN]
    if len(active_chars) == 10:
        score += 15 # Prefer the "perfect" length for standard plates

    return score

def consolidate_ocr_results(plate_data, checksum_exclude=None, validate_indian_plate=True):
    """
    Advanced Positional Voting consolidation with Batch confidence scoring.
    plate_data: List of (text, confidence) tuples OR List of strings (legacy).
    """
    if not plate_data:
        return [None, []], 0
        
    # Standardize input to list of (text, confidence)
    if isinstance(plate_data[0], str):
        voter_input = [(s, 1.0) for s in plate_data]
    else:
        voter_input = plate_data
        
    voter_data = []
    for s, conf in voter_input:
        clean = normalize(s)
        if not clean: continue
        aligned, score = align_to_template(clean)
        voter_data.append({"list": aligned, "score": score, "conf": conf, "text": clean})
        
    if not voter_data:
        return [None, []], 0

    # 1. Generate Grouping Summary for Diagnostics
    group_map = {}
    for ev in voter_data:
        t = ev["text"]
        if t not in group_map:
            group_map[t] = {"count": 0, "conf_sum": 0, "score": ev["score"]}
        group_map[t]["count"] += 1
        group_map[t]["conf_sum"] += ev["conf"]
        
    group_summary = []
    for text, info in group_map.items():
        group_summary.append({
            "text": text,
            "count": info["count"],
            "avg_conf": round(info["conf_sum"] / info["count"], 3),
            "score": info["score"]
        })
    # Sort by count (primary) and structural score (secondary)
    group_summary.sort(key=lambda x: (x["count"], x["score"]), reverse=True)

    # 2. 11-slot Positional Voting
    reconstruction = []
    
    # Pre-calculate weights for every candidate frame to avoid redundant math
    for ev in voter_data:
        # Exponential Weighting: Score 100 (5.00) vs 80 (4.00) is a 10x difference in trust.
        # This ensuring a high-quality frame massively outweighs noisy fragments.
        ev["weight"] = (10.0 ** (ev["score"] / 20.0)) * ev["conf"]
        
        # Fragment Penalty: If very short, reduce its total influence
        if len([c for c in ev["list"] if c != EMPTY_TOKEN]) < 6:
            ev["weight"] *= 0.1
            
    for pos in range(11):
        votes = {}
        for ev in voter_data:
            char = ev["list"][pos]
            weight = ev["weight"]
            
            # Boost real characters over EMPTY_TOKEN during position voting
            if char != EMPTY_TOKEN:
                weight *= 1.5
            
            votes[char] = votes.get(char, 0) + weight
        
        if votes:
            winner = max(votes, key=votes.get)
            # If EMPTY wins but a real character has significant support (>30%), prefer the character
            if winner == EMPTY_TOKEN:
                eligible = {k: v for k, v in votes.items() if k != EMPTY_TOKEN}
                if eligible:
                    best_char = max(eligible, key=eligible.get)
                    total_votes = sum(votes.values())
                    if total_votes > 0 and eligible[best_char] / total_votes > 0.3:
                        winner = best_char
            
            if winner != EMPTY_TOKEN:
                reconstruction.append(winner)
    
    final_plate = "".join(reconstruction)
    
    # 3. Final Confidence Extraction (Confidence of the winning group)
    final_conf = group_summary[0]["avg_conf"] if group_summary else 0.0
    
    if validate_indian_plate:
        # Final Format check
        is_valid = bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', final_plate))
        if not is_valid and re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', final_plate):
            is_valid = True
                
        if not is_valid:
            logging.info(f"[OCR] Rejected reconstruction: {final_plate}")
            return [None, group_summary, 0.0], 0
            
        return [final_plate, group_summary, final_conf], 1
    
    return [final_plate, group_summary, final_conf], 0

# --- Legacy Compatibility Shims ---
def apply_indian_corrections(plate):
    aligned, _ = align_to_template(plate)
    return "".join([c for c in aligned if c != EMPTY_TOKEN])

def is_valid_indian_plate(plate):
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', plate)) or \
           bool(re.match(r'^[0-9]{2}BH[0-9]{4}[A-Z]{2}$', plate))
