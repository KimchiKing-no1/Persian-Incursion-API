# ──────────────────────────────────────────────────────────────
# Step 1 : GAME DATA (UPDATED WITH DETAILED INFO FROM RULEBOOKS)
# ──────────────────────────────────────────────────────────────

# Global Constants: Lists of names or generic identifiers.
STRATEGIC_EVENT_RANDOM_FACILITIES = [
    "Natanz Enrichment Plant",
    "Arak Heavy Water Reactor",
    "Esfahan Conversion Facility",
    "Tehran Research Reactor",
    "Bushehr Nuclear Plant",
    "Fuel Manufacturing Plant",
    "Parchin Military Complex",
    "Fordow Enrichment Site",
    "Qom Uranium Enrichment Facility",
    "Abadan Oil Refinery",
    "Kharg Island Oil Terminal",
    "Arak Oil Refinery", # Added for generic refinery example
    "Kermanshah Oil Refinery", # Added from generic refinery list
    "Tabriz Oil Refinery", # Added from generic refinery list
    "Tehran Oil Refinery", # Added from generic refinery list
    "Shiraz Oil Refinery", # Added from generic refinery list
    "Bandar Abbas Oil Refinery", # Added from generic refinery list
    "Isfahan Oil Refinery", # Added from generic refinery list
    "Lavan Island Oil Refinery", # Added from generic refinery list
    "Sirri Island Oil Terminal", # Added from generic terminal list
    "Lavan Island Oil Terminal", # Added from generic terminal list
    "Ras Bahregan Oil Terminal", # Added from generic terminal list
    "Neka Oil Terminal", # Added from generic terminal list
    # Tactical Air Bases (TABs) - These are locations, not just targets.
    "TAB 1 Mehrabad", "TAB 2 Tabriz", "TAB 3 Shahrokhi (Hamadan)", "TAB 4 Vahdati (Dezful)",
    "TAB 5 Omidiyeh", "TAB 6 Bushehr Airport", "TAB 7 Shiraz", "TAB 8 Khatami (Isfahan)",
    "TAB 9 Bandar Abbas", "TAB 10 Chahbahar", "TAB 11 Tehran", "TAB 12 Tehran", "TAB 13 Zahedan", "TAB 14 Mashhad" # From map on page 28
]

STRATEGIC_EVENT_RANDOM_COUNTRIES = [
    "US", "Russia", "China", "Saudi Arabia/GCC", "Turkey", "Jordan", "UN"
]

# Strategic Events (Details from PersianIncursionRuleBook.pdf page 1)
STRATEGIC_EVENTS = {
    1: {
        "name": "Domestic Scandal",
        "effect": "Lose half Political Points (min 3 PP). Opinion of all other countries (and own domestic) drops by 1. Can't happen if no overt attack in last 3 turns.",
        "trigger": "Player's own roll."
    },
    2: {
        "name": "Freak Weather",
        "effect": "No aircraft may operate in a randomly determined sector for the Map Turn. (If Iran rolls, affects Israel; if Israel rolls, affects Iran's sector).",
        "trigger": "Player's own roll."
    },
    3: {
        "name": "Industrial Accident",
        "effect": "Random Iranian nuclear facility suffers an accident (explosion/radiation). Iranian domestic opinion drops by 3. Israel rolls 1 opinion die on Turkey, Jordan, Saudi Arabia/GCC, and UN.",
        "trigger": "Player's own roll (Iran)." # Implied to be Iran specific from effect.
    },
    4: {
        "name": "Military Accident",
        "effect": "Player's military suffers major accident. Lose 5 Military Points. Opponent rolls 1 opinion die on player's domestic opinion track.",
        "trigger": "Player's own roll."
    },
    5: {
        "name": "Bad Targeting",
        "effect": "Accidentally strike a noncombatant. Lose 5 Political Points. Opponent rolls 1 opinion die on affected country's opinion track and UN. (Also triggered by Urban Target ballistic missile hit of '01').",
        "trigger": "Player's own roll OR Iranian ballistic missile '01' urban hit."
    },
    6: {
        "name": "Political Pressure",
        "effect": "Lose 2 Military Points for D6/3 days (round up).",
        "trigger": "Player's own roll."
    },
    7: {
        "name": "War Hero Scandal",
        "effect": "Lose 5 Political Points.",
        "trigger": "Player's own roll."
    },
    8: {
        "name": "Espionage Arrest",
        "effect": "Opponent loses 5 Intelligence Points. Player rolls 1 opinion die on opponent's domestic opinion track.",
        "trigger": "Player's own roll."
    },
    9: {
        "name": "Third-Party Troubles",
        "effect": "Random third-party country's opinion resets to 0 (Neutral).",
        "trigger": "Player's own roll."
    },
    10: {
        "name": "Intifada Erupts",
        "effect": "Iran rolls 2 opinion dice on Israel's domestic opinion track. Iran can spend up to 12 MP (3 MP per additional die) to add more dice to this roll.",
        "trigger": "Player's own roll (Iran)." # Implied to be Iran specific from effect.
    },
}


# Card Definitions: Standardized structure for easier parsing.
# The 'Effect' field is kept as raw text for now, but in a full simulation,
# it would likely be parsed into structured commands.
# 'Cost' uses a consistent format (e.g., "1P", "5M", "2P, 2I, 2M").
# 'Quantity' indicates how many copies of the card are in the deck.
# 'Notes' captures special rules like 'Backfire', 'Requires', 'Dirty', 'Covert'.

IRAN_CARDS = [
    {"No": 1, "Name": "Ahmadinejad Speech", "Quantity": 3, "Cost": "1P", "Effect": "3● Ir Domest, Is Domestic, SA, J, T, R, C, US", "Notes": "Backfire 7 - Opponent Rolls"},
    {"No": 2, "Name": "Antiwar Protests", "Quantity": 2, "Cost": "6P", "Effect": "4●+4● Is Domestic, US, UN", "Notes": ""},
    {"No": 3, "Name": "Appeal to the Faithful", "Quantity": 1, "Cost": "3P", "Effect": "1● Ir Domestic", "Notes": ""},
    {"No": 4, "Name": "Arms Purchase", "Quantity": 1, "Cost": "5M", "Effect": "Immediately receive any combination of upgrades costing up to 40 points", "Notes": "Supplying nation must be Supporter or Ally"},
    {"No": 5, "Name": "Black Market", "Quantity": 2, "Cost": "--", "Effect": "For every three points spent receive 1 point of any type the player chooses", "Notes": "+1 point of retrieved card’s first cost"},
    {"No": 6, "Name": "Careful Planning", "Quantity": 3, "Cost": "2P", "Effect": "Retrieve 1 card from discard", "Notes": ""},
    {"No": 7, "Name": "Collateral Damage", "Quantity": 3, "Cost": "2P", "Effect": "1●+1●+1● SA, J, T, R, US, UN", "Notes": "Requires: Israel has launched an airstrike this Map Turn"},
    {"No": 8, "Name": "Crackdown on Dissent", "Quantity": 2, "Cost": "2M", "Effect": "1● Ir Domestic", "Notes": "Backfire 8 - Opponent Rolls"},
    {"No": 9, "Name": "Carnage for the Cameras", "Quantity": 1, "Cost": "4P", "Effect": "1●+1● SA, J, T, R, US, UN", "Notes": "Dirty"},
    {"No": 10, "Name": "Fanning the Flames", "Quantity": 2, "Cost": "4P", "Effect": "1● SA, J, T, UN + 1● UN", "Notes": ""},
    {"No": 11, "Name": "Friendship Prices", "Quantity": 2, "Cost": "4P", "Effect": "3●+3● UN, C", "Notes": ""},
    {"No": 12, "Name": "Funding Opposition", "Quantity": 1, "Cost": "3I", "Effect": "3● Is Domestic", "Notes": "Covert"},
    {"No": 13, "Name": "High-Level Indecision", "Quantity": 1, "Cost": "4I", "Effect": "Discard a card from opponent’s River", "Notes": ""},
    {"No": 14, "Name": "Human Interest Story", "Quantity": 1, "Cost": "2P", "Effect": "3● IS, US, UN, SA, J", "Notes": ""},
    {"No": 15, "Name": "Illicit Bribery", "Quantity": 1, "Cost": "5I", "Effect": "Discard 2 cards (random) from opponent’s River", "Notes": "Dirty"},
    {"No": 16, "Name": "Incriminating Photos", "Quantity": 2, "Cost": "4P", "Effect": "4● Ir Domestic, J, SA, T", "Notes": "Dirty"},
    {"No": 17, "Name": "International Divestment", "Quantity": 1, "Cost": "2P", "Effect": "3● US, UN", "Notes": ""},
    {"No": 18, "Name": "Israeli Condemnation", "Quantity": 1, "Cost": "7P", "Effect": "7● Is Domestic", "Notes": ""},
    {"No": 19, "Name": "Major Expose", "Quantity": 2, "Cost": "3P", "Effect": "4●+4●+4● Ir Domestic, SA, J, T, R, C, US, UN", "Notes": "Requires last opponent act to be Dirty"},
    {"No": 20, "Name": "Official Coverup", "Quantity": 2, "Cost": "2P, 2I, 2M", "Effect": "Cancel Strategic Event", "Notes": "Covert"},
    {"No": 21, "Name": "On-Call Kill Team", "Quantity": 1, "Cost": "5I", "Effect": "Cancel opponent’s card whose first cost is I", "Notes": "Covert"},
    {"No": 22, "Name": "OPEC Diplomacy", "Quantity": 1, "Cost": "5P", "Effect": "3●+3● SA, J, C", "Notes": ""},
    {"No": 23, "Name": "Palestinian Unrest", "Quantity": 2, "Cost": "3P", "Effect": "1● SA, J, T + 1D 3● Is Domestic", "Notes": "Dirty"},
    {"No": 24, "Name": "Plausible Deniability", "Quantity": 1, "Cost": "3P", "Effect": "4● SA, J, T or 3● Is Domestic", "Notes": "Dirty. Requires: Must have captured at least one Israeli POW"},
    {"No": 25, "Name": "POWs on TV", "Quantity": 2, "Cost": "4P", "Effect": "3●+3● Is Domestic, Ir Domestic, UN", "Notes": ""},
    {"No": 26, "Name": "Press Leak", "Quantity": 2, "Cost": "3P", "Effect": "3●+3●+3● SA, J, T, R, C, US, UN", "Notes": "Requires: Last opponent act was Covert or Dirty"},
    {"No": 27, "Name": "Promised Concessions", "Quantity": 1, "Cost": "4P", "Effect": "3●+3● R, C, US", "Notes": ""},
    {"No": 28, "Name": "Propaganda Barrage", "Quantity": 2, "Cost": "4P", "Effect": "3●+3● SA, J, T", "Notes": ""},
    {"No": 29, "Name": "Protests in Tel Aviv", "Quantity": 1, "Cost": "5P", "Effect": "4● Is Domestic", "Notes": ""},
    {"No": 30, "Name": "Quick Spin Control", "Quantity": 1, "Cost": "6P", "Effect": "Cancel opponent’s card whose first cost is P", "Notes": ""},
    {"No": 31, "Name": "Radical Elements", "Quantity": 1, "Cost": "4P", "Effect": "3●+3● SA, J, T", "Notes": "Backfire 10 - Opponent Rolls"},
    {"No": 32, "Name": "Sleeper Agent", "Quantity": 2, "Cost": "4I", "Effect": "Look at opponent’s River", "Notes": ""},
    {"No": 33, "Name": "Staged Protest", "Quantity": 1, "Cost": "4I", "Effect": "3● IS, UN", "Notes": ""}, # Note: 'IS' likely means 'Israeli Domestic', confirmed from rulebook context.
    {"No": 34, "Name": "Superpower Pressure", "Quantity": 1, "Cost": "4P", "Effect": "5● SA, T", "Notes": "Requires either R or C Supporter or Ally"}
]
IRAN_CARD_MAP = {c["No"]: c for c in IRAN_CARDS}

ISRAEL_CARDS = [
    {"No": 1, "Name": "Appeal to the Electorate", "Quantity": 1, "Cost": "3P", "Effect": "1● Is Domestic", "Notes": ""},
    {"No": 2, "Name": "Arms Purchase", "Quantity": 1, "Cost": "5M", "Effect": "Immediately receive any combination of upgrades costing up to 40 points", "Notes": "Supplying nation must be Supporter or Ally"},
    {"No": 3, "Name": "Black Market", "Quantity": 2, "Cost": "--", "Effect": "For every three points spent receive 1 point of any type the player chooses", "Notes": "+1 point of retrieved card’s first cost"},
    {"No": 4, "Name": "Illicit Bribery", "Quantity": 1, "Cost": "5I", "Effect": "Discard 2 cards (random) from opponent’s River", "Notes": "Dirty"},
    {"No": 5, "Name": "Careful Planning", "Quantity": 2, "Cost": "2P", "Effect": "Retrieve 1 card from discard", "Notes": ""},
    {"No": 6, "Name": "Collective Anxiety", "Quantity": 1, "Cost": "4P", "Effect": "4●+4● SA, J, T, Is domestic", "Notes": ""},
    {"No": 7, "Name": "Congressional Lobby", "Quantity": 1, "Cost": "2P", "Effect": "1● US", "Notes": ""},
    {"No": 8, "Name": "Cruiser Deployment", "Quantity": 1, "Cost": "4M", "Effect": "Add Bunker Hull-class GC to ballistic Missile Defenses", "Notes": "US is Supporter or Ally"},
    {"No": 9, "Name": "Destroyer Deployment", "Quantity": 1, "Cost": "3M", "Effect": "Add Burke-class DDG to ballistic Missile Defenses", "Notes": "US is Supporter or Ally"},
    {"No": 10, "Name": "Fallout Zone Anxiety", "Quantity": 1, "Cost": "4P", "Effect": "4● SA, J, T", "Notes": ""},
    {"No": 11, "Name": "Fanning the Flames", "Quantity": 1, "Cost": "4P", "Effect": "1● SA, J, T, UN + 1● UN", "Notes": ""},
    {"No": 12, "Name": "Favorable Media", "Quantity": 2, "Cost": "4P", "Effect": "4●+4● US, R, UN, Is Domestic", "Notes": ""},
    {"No": 13, "Name": "Firm Commitment", "Quantity": 2, "Cost": "5P", "Effect": "Choose one of the three required countries. The country’s Opinion Track cannot be moved at all for D6+3 Map Turns", "Notes": ""},
    {"No": 14, "Name": "Flyover Negotiations", "Quantity": 3, "Cost": "5P", "Effect": "5● SA, T", "Notes": ""},
    {"No": 15, "Name": "Focused Diplomacy", "Quantity": 2, "Cost": "4P", "Effect": "3● SA, J, T, US, R, C, UN", "Notes": ""},
    {"No": 16, "Name": "Funding Opposition", "Quantity": 1, "Cost": "3I", "Effect": "3● Is Domestic", "Notes": "Covert"},
    {"No": 17, "Name": "Gulfside Negotiations", "Quantity": 2, "Cost": "4P", "Effect": "4● SA, J", "Notes": ""},
    {"No": 18, "Name": "High-Level Indecision", "Quantity": 1, "Cost": "4I", "Effect": "Discard a card from opponent’s River", "Notes": ""},
    {"No": 19, "Name": "Incriminating Photos", "Quantity": 2, "Cost": "4P", "Effect": "4● Ir Domestic, J, SA, T", "Notes": "Dirty"},
    {"No": 20, "Name": "Israeli Outrage", "Quantity": 2, "Cost": "3P", "Effect": "3● Is Domestic", "Notes": "Requires Iranian Overt act"},
    {"No": 21, "Name": "Long-Term Interests", "Quantity": 1, "Cost": "5P", "Effect": "3●+3●+3● SA, US, C, UN", "Notes": ""},
    {"No": 22, "Name": "Major Expose", "Quantity": 2, "Cost": "3P", "Effect": "4●+4●+4● Ir Domestic, SA, J, T, R, C, US, UN", "Notes": "Requires last opponent act to be Dirty"},
    {"No": 23, "Name": "Manufactured Attack", "Quantity": 1, "Cost": "4I", "Effect": "4● SA, J, T, UN, Is Domestic", "Notes": "Dirty"},
    {"No": 24, "Name": "Media Reaction", "Quantity": 1, "Cost": "3P", "Effect": "3●+3● US, R, UN", "Notes": "Requires previous Iranian Overt act"},
    {"No": 25, "Name": "Nuclear Proof", "Quantity": 1, "Cost": "3P, 3I", "Effect": "3●+3● SA, J, T, US, R, C, US, UN, Is Domestic", "Notes": "Requires at least 1 nuclear target destroyed or crippled"},
    {"No": 26, "Name": "Official Coverup", "Quantity": 2, "Cost": "2P, 2I, 2M", "Effect": "Cancel Strategic Event", "Notes": ""},
    {"No": 27, "Name": "On-Call Kill Team", "Quantity": 1, "Cost": "5I", "Effect": "Cancel opponent’s card whose first cost is I", "Notes": "Covert"},
    {"No": 28, "Name": "Overzealous Pasdaran", "Quantity": 1, "Cost": "3P", "Effect": "4●+4● SA, J, T, US, UN, Is Domestic", "Notes": "Add Cost 3I. Requires previous Iranian Overt"},
    {"No": 29, "Name": "Plausible Deniability", "Quantity": 1, "Cost": "5I", "Effect": "4● SA, J, T or 3● Ir Domestic", "Notes": "Dirty"},
    {"No": 30, "Name": "Press Leak", "Quantity": 2, "Cost": "3P", "Effect": "3●+3●+3● SA, J, T, R, C, US, UN", "Notes": "Requires: Last opponent act was Covert or Dirty"},
    {"No": 31, "Name": "Protests in Tehran", "Quantity": 2, "Cost": "5P", "Effect": "4● Ir Domestic", "Notes": ""},
    {"No": 32, "Name": "Quick Spin Control", "Quantity": 1, "Cost": "6P", "Effect": "Cancel opponent’s card whose first cost is P", "Notes": ""},
    {"No": 33, "Name": "Sleeper Agent", "Quantity": 2, "Cost": "4I", "Effect": "Look at opponent’s River", "Notes": ""},
    {"No": 34, "Name": "Speech of Support", "Quantity": 2, "Cost": "3P", "Effect": "3● US, Is domestic", "Notes": ""},
    {"No": 35, "Name": "Technology Transfer", "Quantity": 1, "Cost": "4M", "Effect": "3● C, R", "Notes": "Backfire – Ir receives 3● US."},
    {"No": 36, "Name": "UN Resolution", "Quantity": 1, "Cost": "1P", "Effect": "1● UN, Is Domestic", "Notes": "Israel still 3● C, R, dice if Backfire is rolled."}, # Typo fixed from 'Resoulution'
    {"No": 37, "Name": "Widespread Condemnation", "Quantity": 1, "Cost": "7P", "Effect": "7● Ir Domestic", "Notes": "Israel still 3● C, R, dice if Backfire is rolled."},
    {"No": 38, "Name": "Worried Leaders", "Quantity": 1, "Cost": "7P", "Effect": "4●+4●+4● SA,J,T,R,C,UN, US", "Notes": ""} # Typo fixed from ',J,'
]
ISRAEL_CARD_MAP = {c["No"]: c for c in ISRAEL_CARDS}


# SAM_COMBAT_TABLE: Detailed data for SAM systems.
# 'Hit_Chance_Aircraft': Dictionary with specific hit chances per aircraft type.
# 'Attacks_Per_Battery': 'X' (fixed) or 'X/Y' (per vehicle/total for mobile).
# 'Outbound_Attacks': Boolean indicating if it can attack outbound raids.
# 'Optical_Backup': Boolean.
# 'Range_nm': Engagement range in nautical miles.
# 'Advanced_SAM': Boolean, for special rules like suppression.
# 'Long_Range_Node_Capable': Boolean (S-300, HQ-9 can engage in both long and medium).
SAM_COMBAT_TABLE = {
    # SAM Type: {Aircraft_Type: Hit_Chance_Decimal, 'Attacks_Per_Battery': 'X/Y' or 'X', 'Outbound_Attacks': bool, 'Optical_Backup': bool, 'Range_nm': float}
    # Data derived from "SAM Combat Table" in PersianIncursionRuleBook.pdf (page 34)
    "Buk-M1 [SA-11]": {
        "Hit_Chance_Aircraft": {"F-15I": 0.35, "F-16I": 0.45, "EA-18G": 0.30, "F/A-18": 0.75, "F-22A": 0.02, "B-2": 0.50, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': True, 'Optical_Backup': True, 'Range_nm': 18.9, 'Advanced_SAM': True, 'Long_Range_Node_Capable': False
    },
    "HQ-2J/Sayyad": {
        "Hit_Chance_Aircraft": {"F-15I": 0.05, "F-16I": 0.05, "EA-18G": 0.05, "F/A-18": 0.30, "F-22A": 0.02, "B-2": 0.05, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': False, 'Optical_Backup': False, 'Range_nm': 16.2, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "HQ-7 [FM-80]/Shahab Taqeb": {
        "Hit_Chance_Aircraft": {"F-15I": 0.15, "F-16I": 0.20, "EA-18G": 0.10, "F/A-18": 0.60, "F-22A": 0.02, "B-2": 0.30, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '3', 'Outbound_Attacks': False, 'Optical_Backup': True, 'Range_nm': 4.6, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "HQ-9": {
        "Hit_Chance_Aircraft": {"F-15I": 0.45, "F-16I": 0.50, "EA-18G": 0.35, "F/A-18": 0.80, "F-22A": 0.02, "B-2": 0.60, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '6', 'Outbound_Attacks': True, 'Optical_Backup': False, 'Range_nm': 54.0, 'Advanced_SAM': True, 'Long_Range_Node_Capable': True
    },
    "I-Hawk/Shahin": {
        "Hit_Chance_Aircraft": {"F-15I": 0.15, "F-16I": 0.20, "EA-18G": 0.10, "F/A-18": 0.60, "F-22A": 0.02, "B-2": 0.30, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': False, 'Optical_Backup': False, 'Range_nm': 21.6, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "Kub-M3 [SA-6 Gainful]": {
        "Hit_Chance_Aircraft": {"F-15I": 0.10, "F-16I": 0.15, "EA-18G": 0.05, "F/A-18": 0.50, "F-22A": 0.02, "B-2": 0.10, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': True, 'Optical_Backup': True, 'Range_nm': 13.5, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "Pantsyr-S1E [SA-22 Greyhound]": {
        "Hit_Chance_Aircraft": {"F-15I": 0.60, "F-16I": 0.65, "EA-18G": 0.60, "F/A-18": 0.90, "F-22A": 0.15, "B-2": 0.70, "PGMs_HARMs": 0.90, "PGMs": 0.75}, # Added PGM hit chance
        'Attacks_Per_Battery': '2/12', 'Outbound_Attacks': True, 'Optical_Backup': True, 'Range_nm': 11.0, 'Advanced_SAM': True, 'Long_Range_Node_Capable': False
    },
    "QW-2/Misagh-2": { # MANPADS
        "Hit_Chance_Aircraft": {"F-15I": 0.15, "F-16I": 0.20, "EA-18G": 0.10, "F/A-18": 0.45, "F-22A": 0.02, "B-2": 0.25, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': False, 'Optical_Backup': True, 'Range_nm': 2.7, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "Rapier": {
        "Hit_Chance_Aircraft": {"F-15I": 0.20, "F-16I": 0.30, "EA-18G": 0.15, "F/A-18": 0.65, "F-22A": 0.02, "B-2": 0.35, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '6', 'Outbound_Attacks': False, 'Optical_Backup': True, 'Range_nm': 3.5, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "RBS-70": { # MANPADS
        "Hit_Chance_Aircraft": {"F-15I": 0.25, "F-16I": 0.30, "EA-18G": 0.20, "F/A-18": 0.55, "F-22A": 0.02, "B-2": 0.45, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': False, 'Optical_Backup': True, 'Range_nm': 4.5, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False
    },
    "S-200 [SA-5 Gammon] Ghareh": {
        "Hit_Chance_Aircraft": {"F-15I": 0.05, "F-16I": 0.05, "EA-18G": 0.05, "F/A-18": 0.30, "F-22A": 0.02, "B-2": 0.05, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '1', 'Outbound_Attacks': False, 'Optical_Backup': False, 'Range_nm': 135.0, 'Advanced_SAM': False, 'Long_Range_Node_Capable': False # S-200 cannot engage stealth
    },
    "S-300PMU-1 [SA-20a Gargoyle]": {
        "Hit_Chance_Aircraft": {"F-15I": 0.45, "F-16I": 0.60, "EA-18G": 0.35, "F/A-18": 0.80, "F-22A": 0.02, "B-2": 0.55, "PGMs_HARMs": 0.0},
        'Attacks_Per_Battery': '6', 'Outbound_Attacks': True, 'Optical_Backup': False, 'Range_nm': 81.0, 'Advanced_SAM': True, 'Long_Range_Node_Capable': True
    },
    "Tor-M1 [SA-15]": {
        "Hit_Chance_Aircraft": {"F-15I": 0.45, "F-16I": 0.50, "EA-18G": 0.35, "F/A-18": 0.80, "F-22A": 0.02, "B-2": 0.60, "PGMs_HARMs": 0.85, "PGMs": 0.65}, # Added PGM hit chance
        'Attacks_Per_Battery': '2/8', 'Outbound_Attacks': True, 'Optical_Backup': True, 'Range_nm': 6.5, 'Advanced_SAM': True, 'Long_Range_Node_Capable': False
    },
}

# AAA_COMBAT_TABLE: Lookup table for Anti-Aircraft Artillery hits.
# Key: AA_Value, Value: Dictionary of {2D6_Roll: Hits}.
AAA_COMBAT_TABLE = {
    0.0: {r: 0 for r in range(2, 13)}, # Not explicitly in rulebook, but logical for 0 AA
    0.2: {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:1}, # Estimated for single 23mm from 0.2*0.25=0.05 base strength / gun
    0.3: {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:1,11:1,12:1}, # Estimated for single 35mm
    0.5: {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:1,12:1},
    0.6: {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:1,10:1,11:1,12:1},
    0.75: {2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:1,10:1,11:1,12:2},
    1.0: {2:0,3:0,4:0,5:0,6:0,7:0,8:1,9:1,10:1,11:1,12:2},
    1.25: {2:0,3:0,4:0,5:0,6:0,7:1,8:1,9:1,10:1,11:1,12:2},
    1.5: {2:0,3:0,4:0,5:0,6:1,7:1,8:1,9:1,10:2,11:2,12:3},
    1.75: {2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:2,10:2,11:3,12:3},
    1.8: {2:0,3:0,4:1,5:1,6:1,7:1,8:2,9:2,10:3,11:3,12:4}, # As used for Natanz
    2.0: {2:0,3:0,4:1,5:1,6:1,7:1,8:2,9:2,10:3,11:3,12:4},
    2.5: {2:0,3:0,4:1,5:1,6:1,7:2,8:2,9:3,10:3,11:4,12:4}, # As used for Isfahan
    3.0: {2:0,3:1,4:1,5:1,6:2,7:2,8:3,9:3,10:4,11:4,12:5},
    3.5: {2:0,3:1,4:1,5:1,6:2,7:2,8:3,9:3,10:4,11:5,12:5},
    4.0: {2:0,3:1,4:1,5:1,6:2,7:2,8:3,9:4,10:4,11:5,12:6},
    4.5: {2:1,3:1,4:1,5:1,6:2,7:2,8:3,9:4,10:4,11:5,12:6},
    5.0: {2:1,3:1,4:1,5:2,6:2,7:3,8:4,9:4,10:5,11:5,12:6},
    5.5: {2:1,3:1,4:1,5:2,6:2,7:3,8:4,9:5,10:5,11:6,12:6},
    6.0: {2:1,3:1,4:2,5:2,6:2,7:3,8:4,9:5,10:6,11:6,12:7},
    6.5: {2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:7},
    7.0: {2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:7},
    7.5: {2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:7,12:7},
    8.0: {2:1,3:1,4:2,5:3,6:3,7:4,8:4,9:5,10:6,11:7,12:8},
    8.5: {2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:6,11:7,12:8},
    9.0: {2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:6,11:7,12:8},
}

# PGM_ATTACK_TABLE: Ordnance data for Israeli attacks.
# Source: PersianIncursionRuleBook.pdf (page 35)
PGM_ATTACK_TABLE = {
    "AGM-88 HARM": {"Guidance": "ARM", "Generation": 3, "Hits": 3, "Armor_Pen": None, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.08, "G": 0.08}, "Range_nm": 70, "Notes": "Destroys radar"},
    "EGBU-28B": {"Guidance": "GPS/Laser", "Generation": "1/3", "Hits": 2, "Armor_Pen": 44, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.07, "F": 0.06, "G": 0.04}, "Range_nm": 11, "Notes": ""},
    "EGBU-28C": {"Guidance": "GPS/Laser", "Generation": "1/3", "Hits": 2, "Armor_Pen": 53, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.07, "G": 0.04}, "Range_nm": 11, "Notes": "Can penetrate Natanz halls"},
    "GBU-31 JDAM": {"Guidance": "GPS", "Generation": 1, "Hits": 2, "Armor_Pen": 5, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.07, "E": 0.08, "F": 0.06, "G": 0.04}, "Range_nm": 13, "Notes": ""},
    "GBU-32 JDAM": {"Guidance": "GPS", "Generation": 1, "Hits": 1, "Armor_Pen": 5, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.07, "F": 0.06, "G": 0.04}, "Range_nm": 13, "Notes": ""},
    "GBU-38 JDAM": {"Guidance": "GPS", "Generation": 1, "Hits": 1, "Armor_Pen": 5, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.07, "E": 0.08, "F": 0.06, "G": 0.04}, "Range_nm": 13, "Notes": ""},
    "GBU-39 SDB": {"Guidance": "GPS", "Generation": 2, "Hits": 1, "Armor_Pen": 9, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.07, "G": 0.06}, "Range_nm": 60, "Notes": ""},
    "GBU-57 MOP": {"Guidance": "GPS", "Generation": 2, "Hits": 3, "Armor_Pen": 105, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.07, "G": 0.06}, "Range_nm": 7, "Notes": "Penetrates all layers"},
    "Guillotine": {"Guidance": "Laser", "Generation": 3, "Hits": 2, "Armor_Pen": 5, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.07, "G": 0.05}, "Range_nm": 13, "Notes": ""},
    "Spice 2000": {"Guidance": "GPS&EO", "Generation": "1&2", "Hits": 2, "Armor_Pen": 7, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.07, "G": 0.06}, "Range_nm": 32, "Notes": ""},
    "STAR-1": {"Guidance": "ARM", "Generation": 3, "Hits": 3, "Armor_Pen": None, "Hit_Chance_Target_Size": {"A": 0.08, "B": 0.08, "C": 0.08, "D": 0.08, "E": 0.08, "F": 0.08, "G": 0.08}, "Range_nm": 135, "Notes": "Destroys radar"},
}

# TARGET_DEFENSES: Comprehensive structure for all targets.
# Nested dictionaries for clear organization.
# 'Long_Range_SAMs', 'Medium_Range_SAMs', 'Short_Range_SAMs': Lists of SAM types present.
# 'AAA_Value': Overall AA strength.
# 'AAA_Guns': Dictionary of specific AAA gun counts (for initial setup and specific rule checks).
# 'MANPADS_positions': Number of MANPADS positions.
# 'Armor_Class': Dictionary of armor values for different components.
# 'Target_Types': List of categories (e.g., "Nuclear", "Oil_Refinery", "Military").
# 'Primary_Targets', 'Secondary_Targets': Dictionaries of components, with initial 'damage_boxes_hit' and 'max_damage_for_crippled'/'max_damage_for_destroyed'.
TARGET_DEFENSES = {
    "Natanz Uranium Enrichment Facility": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad", "Kub-M3 [SA-6 Gainful]"],
        "Short_Range_SAMs": ["Tor-M1 [SA-15]"],
        "AAA_Value": 1.8,
        "AAA_Guns": {"Mod KS-19 100mm battery": 10, "Skyguard site": 4, "Type 85 23mm and GDF 35mm AA gun emplacement": 25},
        "MANPADS_positions": 4,
        "Armor_Class": {"Underground Centrifuge Hall": 48, "UF6 Storage": 48, "Building covering tunnel entrance": 7, "Other": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {
            "Original Centrifuge Plant - Centrifuge Assembly complex": {"damage_boxes_hit": 0, "max_damage_for_crippled": 5, "max_damage_for_destroyed": 9, "armor_class": 0},
            "Underground Facility G1-4 Centrifuge hall": {"damage_boxes_hit": 0, "max_damage_for_crippled": 5, "max_damage_for_destroyed": 9, "armor_class": 48},
            "Underground Facility H1-4 Centrifuge hall": {"damage_boxes_hit": 0, "max_damage_for_crippled": 5, "max_damage_for_destroyed": 9, "armor_class": 48},
            "Underground Facility UF6 Storage": {"damage_boxes_hit": 0, "max_damage_for_crippled": 5, "max_damage_for_destroyed": 9, "armor_class": 48},
            "Building covering tunnel entrance J": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 5, "armor_class": 7},
        },
        "Secondary_Targets": {
            "Steam Plant 1 L": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Steam Plant 2 M": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Air Handling Building N1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Air Handling Building N2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Transformer Station P1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Transformer Station P2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Transformer Station P3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Transformer Station P4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Backup Gas Turbine Generator Q1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Backup Gas Turbine Generator Q2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Transformer Substation R": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
        }
    },
    "Arak Heavy Water Reactor": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Short_Range_SAMs": ["Tor-M1 [SA-15]"],
        "AAA_Value": 1.8,
        "AAA_Guns": {"Skyguard site": 2, "GDF 35mm AAA": 7, "Type 85 23mm AAA": 16},
        "MANPADS_positions": 4,
        "Armor_Class": {"Reactor Dome": 5, "Other": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {
            "Reactor Dome J": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 5},
            "Girdler-Sulfide Unit (Unit 3) A1-6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Girdler-Sulfide Unit (Unit 4) B1-6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Distillation Plant (Prob Unit 5) D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Possible Secondary Loop Support Bldg K": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Building 1 L1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Building 2 L2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Fan Building L3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Possible Hot Cells & Radioisotope Production M": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Reserve Coolant & Pumping Station N": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Spent Fuel Cooling Ponds & Ventilation Stack P": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Cooling Unit C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "H2S Removal Unit and Flare Tower G1-2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Feedwater Handling and Purification H1-3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "H2S Production Plant I": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        }
    },
    "Esfahan Conversion Facility": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Short_Range_SAMs": ["Tor-M1 [SA-15]"],
        "AAA_Value": 2.5,
        "AAA_Guns": {"Skyguard site": 5, "GDF 35mm AAA": 7, "Type 85 23mm AAA": 14},
        "MANPADS_positions": 4,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {
            "Uranium Conversion Facility A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Uranium Conversion Facility A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Uranium Conversion Facility A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Uranium Conversion Facility A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
        },
        "Secondary_Targets": {
            "UCF Support C1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C7": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C8": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C9": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C10": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C11": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C12": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "UCF Support C13": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        }
    },
    "Tehran Research Reactor": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": [],
        "AAA_Value": 0.5,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"Reactor": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {"Reactor": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 3, "armor_class": 0}},
        "Secondary_Targets": {}
    },
    "Bushehr Nuclear Plant": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad"],
        "Short_Range_SAMs": [],
        "AAA_Value": 1.0,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"Plant": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {"Plant": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 5, "armor_class": 0}},
        "Secondary_Targets": {}
    },
    "Fuel Manufacturing Plant": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Short_Range_SAMs": ["Tor-M1 [SA-15]"],
        "AAA_Value": 2.5,
        "AAA_Guns": {"Skyguard site": 5, "GDF 35mm AAA": 7, "Type 85 23mm AAA": 14},
        "MANPADS_positions": 4,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {
            "Fuel Manufacturing Plant B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 5, "armor_class": 0}
        },
        "Secondary_Targets": {
            "Transformer Station D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}
        }
    },
    "Zirconium Production Plant": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Short_Range_SAMs": ["Tor-M1 [SA-15]"],
        "AAA_Value": 2.5,
        "AAA_Guns": {"Skyguard site": 5, "GDF 35mm AAA": 7, "Type 85 23mm AAA": 14},
        "MANPADS_positions": 4,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {
            "Foundry A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Foundry A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Foundry A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Foundry A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Foundry A5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Foundry A6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Fabrication & Finishing B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Fabrication & Finishing B2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Fabrication & Finishing B3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Fabrication & Finishing B4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 0},
            "Fabrication & Finishing B5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Fabrication & Finishing B6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        }
    },
    "Parchin Military Complex": {
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": ["Rapier", "HQ-7 [FM-80]/Shahab Taqeb"],
        "AAA_Value": 1.0,
        "AAA_Guns": {},
        "MANPADS_positions": 2,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military"],
        "Primary_Targets": {"Military Facility": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 5, "armor_class": 0}},
        "Secondary_Targets": {}
    },
    "Fordow Enrichment Site": {
        "Long_Range_SAMs": ["S-300PMU-1 [SA-20a Gargoyle]", "HQ-9"],
        "Medium_Range_SAMs": ["Buk-M1 [SA-11]"],
        "Short_Range_SAMs": ["Tor-M1 [SA-15]", "Pantsyr-S1E [SA-22 Greyhound]"],
        "AAA_Value": 2.5,
        "AAA_Guns": {},
        "MANPADS_positions": 4,
        "Armor_Class": {"Centrifuge Hall": 48},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {"Centrifuge Hall": {"damage_boxes_hit": 0, "max_damage_for_crippled": 3, "max_damage_for_destroyed": 9, "armor_class": 48}},
        "Secondary_Targets": {}
    },
    "Qom Uranium Enrichment Facility": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": [],
        "AAA_Value": 0.5,
        "AAA_Guns": {},
        "MANPADS_positions": 4,
        "Armor_Class": {"Centrifuge Hall": 15},
        "Target_Types": ["Nuclear"],
        "Primary_Targets": {"Centrifuge Hall": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": 15}},
        "Secondary_Targets": {}
    },
    "Abadan Oil Refinery": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": [],
        "AAA_Value": 0.0,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"Power/Steam Plant G1-G2": "Heavy", "Other": 0},
        "Target_Types": ["Oil_Refinery", "Oil_Terminal"],
        "Primary_Targets": {
            "Atmospheric Distillers A1-A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Catalytic Cracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Vacuum Distillers D1-D5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Storage/Pumping I1-I2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Gas Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Power/Steam Plant G1-G2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"},
            "Hydrogen Production Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Tanker Loading Pier J1-J9": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        }
    },
    "Kharg Island Oil Terminal": {
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Short_Range_SAMs": [],
        "AAA_Value": 0.0,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"T Jetty-Stem A7": "Heavy", "Other": 0},
        "Target_Types": ["Oil_Terminal"],
        "Primary_Targets": {
            "T Jetty-Berth Piping A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 4, "max_damage_for_destroyed": 8, "armor_class": 0},
            "T Jetty-Stem A7": {"damage_boxes_hit": 0, "max_damage_for_crippled": 2, "max_damage_for_destroyed": 4, "armor_class": "Heavy"},
            "Pipe Manifold B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Pipe Manifold B2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Sea Island Pumphouse C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Sea Island Pipe Manifold D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Sea Island Pipe Manifold D2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        },
        "Secondary_Targets": {
            "T Jetty-Loading Point A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "T Jetty-Loading Point A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "T Jetty-Loading Point A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "T Jetty-Loading Point A5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "T Jetty-Loading Point A6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Desalination Plant E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Sea Island Berths Piping F1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Sea Island Berths Piping F2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        }
    },
    "Generic Oil Refinery": {
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": [],
        "AAA_Value": 0.0,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0},
        "Target_Types": ["Oil_Refinery"],
        "Primary_Targets": {
            "Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"},
            "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        }
    },
    "Arak Oil Refinery": {
        # Base data from Generic Oil Refinery
        "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}},
        # Arak-specific overrides and additions
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad"],
        "Short_Range_SAMs": [], # Explicitly empty
        "Production": {"barrels_per_day": 170, "percent_total": 0.12},
        "Fighter_Sector": "II"
    },
    "Kermanshah Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad"],
        "Production": {"barrels_per_day": 30, "percent_total": 0.02},
        "Fighter_Sector": "II"
    },
    "Tabriz Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Production": {"barrels_per_day": 100, "percent_total": 0.07},
        "Fighter_Sector": "I"
    },
    "Tehran Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad"],
        "Production": {"barrels_per_day": 220, "percent_total": 0.15},
        "Fighter_Sector": "I"
    },
    "Shiraz Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Production": {"barrels_per_day": 40, "percent_total": 0.03},
        "Fighter_Sector": "III"
    },
    "Bandar Abbas Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad"],
        "Production": {"barrels_per_day": 230, "percent_total": 0.16},
        "Fighter_Sector": "III"
    },
    "Isfahan Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Production": {"barrels_per_day": 250, "percent_total": 0.18},
        "Fighter_Sector": "II"
    },
    "Lavan Island Oil Refinery": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Power/Steam Plant G": "Heavy", "Other": 0}, "Target_Types": ["Oil_Refinery"], "Primary_Targets": {"Atmospheric Distiller A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Hydrocracker B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Catalytic Reformer C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Vacuum Distiller D": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {"Visibreaker E": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "LPG Plant F": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Power/Steam Plant G": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": "Heavy"}, "Hydrogen Plant H": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": ["Type 85 23mm AAA", "GDF 35mm AAA"],
        "AAA_Value": 2.5,
        "AAA_Guns": {"Type 85 23mm AAA": 10, "GDF 35mm AAA": 10},
        "Production": {"barrels_per_day": 30, "percent_total": 0.02},
        "Fighter_Sector": "III"
    },
    "Generic Oil Terminal": {
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": [],
        "AAA_Value": 0.0,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"Tanker Loading Jetty A": 0, "Pumps/Machinery B": 0},
        "Target_Types": ["Oil_Terminal"],
        "Primary_Targets": {
            "Tanker Loading Jetty A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
            "Pumps/Machinery B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0},
        },
        "Secondary_Targets": {}
    },
    "Sirri Island Oil Terminal": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Tanker Loading Jetty A": 0, "Pumps/Machinery B": 0}, "Target_Types": ["Oil_Terminal"], "Primary_Targets": {"Tanker Loading Jetty A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Pumps/Machinery B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Production": {"barrels_per_day": 1454, "percent_total": 0.16},
        "Jetties": 1,
        "Fighter_Sector": "III"
    },
    "Lavan Island Oil Terminal": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Tanker Loading Jetty A": 0, "Pumps/Machinery B": 0}, "Target_Types": ["Oil_Terminal"], "Primary_Targets": {"Tanker Loading Jetty A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Pumps/Machinery B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {}},
        "Long_Range_SAMs": [],
        "Short_Range_SAMs": ["Type 85 23mm AAA", "GDF 35mm AAA"],
        "AAA_Value": 2.5,
        "AAA_Guns": {"Type 85 23mm AAA": 10, "GDF 35mm AAA": 10},
        "Production": {"barrels_per_day": 1086, "percent_total": 0.12},
        "Jetties": 2,
        "Fighter_Sector": "III"
    },
    "Ras Bahregan Oil Terminal": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Tanker Loading Jetty A": 0, "Pumps/Machinery B": 0}, "Target_Types": ["Oil_Terminal"], "Primary_Targets": {"Tanker Loading Jetty A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Pumps/Machinery B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Production": {"barrels_per_day": 1051, "percent_total": 0.11},
        "Jetties": 1,
        "Fighter_Sector": "III"
    },
    "Neka Oil Terminal": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 0.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Tanker Loading Jetty A": 0, "Pumps/Machinery B": 0}, "Target_Types": ["Oil_Terminal"], "Primary_Targets": {"Tanker Loading Jetty A": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}, "Pumps/Machinery B": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 0}}, "Secondary_Targets": {}},
        "Long_Range_SAMs": [],
        "Production": {"barrels_per_day": 298, "percent_total": 0.03},
        "Jetties": 3,
        "Fighter_Sector": "I"
    },
    "Generic Tactical Airbase": {
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": [],
        "Short_Range_SAMs": [],
        "AAA_Value": 1.0,
        "AAA_Guns": {},
        "MANPADS_positions": 0,
        "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7},
        "Target_Types": ["Military", "Airbase"],
        "Primary_Targets": {
            "Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7},
            "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7},
            "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7},
        },
        "Secondary_Targets": {
            "Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "TAB 1 Mehrabad": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin", "HQ-2J/Sayyad"],
        "Fighter_Sector": "I",
        "Capacity": 19,
        "QRS_Count": 2,
        "Magazines_Count": 2,
    },
    "TAB 2 Tabriz": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Fighter_Sector": "I",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 3 Shahrokhi (Hamadan)": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Fighter_Sector": "I",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 4 Vahdati (Dezful)": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Fighter_Sector": "II",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 5 Omidiyeh": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": [],
        "Fighter_Sector": "II",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 6 Bushehr Airport": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Fighter_Sector": "III",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 7 Shiraz": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": [],
        "Fighter_Sector": "II",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 8 Khatami (Isfahan)": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["I-Hawk/Shahin"],
        "Fighter_Sector": "II",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 9 Bandar Abbas": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": ["S-200 [SA-5 Gammon] Ghareh"],
        "Medium_Range_SAMs": ["HQ-2J/Sayyad", "I-Hawk/Shahin"],
        "Fighter_Sector": "III",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "TAB 10 Chahbahar": {
        **{"Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [], "AAA_Value": 1.0, "AAA_Guns": {}, "MANPADS_positions": 0, "Armor_Class": {"Hardened Aircraft Shelter": 7, "Quick Reaction Shelter": 7, "Control Tower": 0, "Magazine": 7}, "Target_Types": ["Military", "Airbase"], "Primary_Targets": {"Hardened Aircraft Shelter A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Quick Reaction Shelter B1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}, "Magazine D1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 2, "armor_class": 7}}, "Secondary_Targets": {"Control Tower C": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0}}},
        "Long_Range_SAMs": [],
        "Medium_Range_SAMs": [],
        "Fighter_Sector": "III",
        "Capacity": 0,
        "QRS_Count": 0,
        "Magazines_Count": 0,
    },
    "Generic Buk-M1 Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "9S18M1 [Snow Drift] acquisition radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "9A310 Fourrail launcher vehicle A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "9A310 Fourrail launcher vehicle A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "9A310 Fourrail launcher vehicle A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "9A310 Fourrail launcher vehicle A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "Generic HQ-2J/Sayyad Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "SJ-202 [Gin Sling] guidance radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Single-rail Launcher A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Single-rail Launcher A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Single-rail Launcher A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Single-rail Launcher A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Single-rail Launcher A5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Single-rail Launcher A6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "Generic HQ-9 Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "HT-233 [Tiger Paw] guidance radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Fourtubed launcher A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Fourtubed launcher A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Fourtubed launcher A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Fourtubed launcher A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Fourtubed launcher A5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Fourtubed launcher A6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "YLC-2V [High Guard] acquisition radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "Generic I-Hawk/Sejil Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "MPQ-46 guidance radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "Triple Launcher A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Triple Launcher A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Triple Launcher A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "MPQ-48 radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "MPQ-50 radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "Generic Kub-M3 Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "1S91 [Straight Flush] guidance radar vehicle": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "2P25 Three-rail launcher vehicle A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "2P25 Three-rail launcher vehicle A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "2P25 Three-rail launcher vehicle A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "2P25 Three-rail launcher vehicle A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Long Track acquisition radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "Generic S-200/Ghareh Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "56N2 [Square Pair] guidance radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "5P72 Single-Rail Launcher A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "5P72 Single-Rail Launcher A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "P-14 [Tall King] radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "PRV13 [Odd Group] radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
    "Generic S-300 Battery": {
        "Long_Range_SAMs": [], "Medium_Range_SAMs": [], "Short_Range_SAMs": [],
        "AAA_Value": 0.2,
        "AAA_Guns": {"Type 85 23mm AAA": 2},
        "MANPADS_positions": 0,
        "Armor_Class": {"All": 0},
        "Target_Types": ["Military", "SAM_Site"],
        "Primary_Targets": {
            "30N6E1 Tomb Stone guidance radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        },
        "Secondary_Targets": {
            "5P85TE Four-tubed TEL A1": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "5P85TE Four-tubed TEL A2": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "5P85TE Four-tubed TEL A3": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "5P85TE Four-tubed TEL A4": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "5P85TE Four-tubed TEL A5": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "5P85TE Four-tubed TEL A6": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "Big Bird acquisition radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
            "76N6 Clam Shell acquisition radar": {"damage_boxes_hit": 0, "max_damage_for_crippled": 1, "max_damage_for_destroyed": 1, "armor_class": 0},
        }
    },
}
VALID_TARGETS = list(TARGET_DEFENSES.keys())
# Initial squadron setups. 'current', 'damaged', 'under_repair', 'lost' will be dynamic state.
# 'RepairLevel' and 'Starting_Strength' are static here.
# Added a 'base_airfield' for clarity on initial deployment.
IRANIAN_SQUADRONS_SETUP = {
    '11 TFS': {'Type': 'MiG-29', 'Starting_Strength': 14, 'RepairLevel': 'second', 'base_airfield': 'TAB 1 Mehrabad'},
    'Detachment 1': {'Type': 'F-14', 'Starting_Strength': 4, 'RepairLevel': 'fourth', 'base_airfield': 'TAB 1 Mehrabad'},
    '12 TFS': {'Type': 'F-5E', 'Starting_Strength': 10, 'RepairLevel': 'third', 'base_airfield': 'TAB 1 Mehrabad'},
    '21 TFS': {'Type': 'F-5E/F', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 2 Tabriz'},
    '22 TFS': {'Type': 'F-5E/F', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 2 Tabriz'},
    '23 TFS': {'Type': 'MiG-29', 'Starting_Strength': 14, 'RepairLevel': 'second', 'base_airfield': 'TAB 2 Tabriz'},
    'Detachment 2': {'Type': 'F-14', 'Starting_Strength': 4, 'RepairLevel': 'fourth', 'base_airfield': 'TAB 2 Tabriz'},
    '32 TFS': {'Type': 'F-4E', 'Starting_Strength': 10, 'RepairLevel': 'third', 'base_airfield': 'TAB 3 Shahrokhi (Hamadan)'},
    '33 TFS': {'Type': 'F-4E', 'Starting_Strength': 10, 'RepairLevel': 'third', 'base_airfield': 'TAB 3 Shahrokhi (Hamadan)'},
    '41 TFS': {'Type': 'F-5E', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 4 Vahdati (Dezful)'},
    '42 TFS': {'Type': 'F-5E', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 4 Vahdati (Dezful)'},
    '43 TFS': {'Type': 'F-5F', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 4 Vahdati (Dezful)'},
    '51 TFS': {'Type': 'F-7M', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 5 Omidiyeh'},
    '52 TFS': {'Type': 'F-7M', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 5 Omidiyeh'},
    '53 TFS': {'Type': 'F-7M', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 5 Omidiyeh'},
    '71 TFS': {'Type': 'F-5E', 'Starting_Strength': 8, 'RepairLevel': 'third', 'base_airfield': 'TAB 7 Shiraz'},
    '81 TFS': {'Type': 'F-14A', 'Starting_Strength': 8, 'RepairLevel': 'fourth', 'base_airfield': 'TAB 8 Khatami (Isfahan)'},
    '82 TFS': {'Type': 'F-14A', 'Starting_Strength': 8, 'RepairLevel': 'fourth', 'base_airfield': 'TAB 8 Khatami (Isfahan)'},
    '83 TFS': {'Type': 'F-14A', 'Starting_Strength': 8, 'RepairLevel': 'fourth', 'base_airfield': 'TAB 8 Khatami (Isfahan)'},
    '61 TFS': {'Type': 'F-4E', 'Starting_Strength': 10, 'RepairLevel': 'third', 'base_airfield': 'TAB 6 Bushehr Airport'},
    '62 TFS': {'Type': 'F-4E', 'Starting_Strength': 10, 'RepairLevel': 'third', 'base_airfield': 'TAB 6 Bushehr Airport'},
    'Detachment 6': {'Type': 'F-14', 'Starting_Strength': 4, 'RepairLevel': 'fourth', 'base_airfield': 'TAB 6 Bushehr Airport'},
    '91 TFS': {'Type': 'F-4E', 'Starting_Strength': 7, 'RepairLevel': 'third', 'base_airfield': 'TAB 9 Bandar Abbas'},
    '92 TFS': {'Type': 'F-4E', 'Starting_Strength': 7, 'RepairLevel': 'third', 'base_airfield': 'TAB 9 Bandar Abbas'},
    '101 TFS': {'Type': 'F-4D', 'Starting_Strength': 5, 'RepairLevel': 'third', 'base_airfield': 'TAB 10 Chahbahar'},
}

ISRAELI_SQUADRONS_SETUP = {
    '69th': {'Type': 'F-15I Ra\'am', 'Starting_Strength': 24, 'RepairLevel': 'first'},
    '107th': {'Type': 'F-16I Sufa', 'Starting_Strength': 24, 'RepairLevel': 'first'},
    '109th F-16I Squadron': {'Type': 'F-16I Sufa', 'Starting_Strength': 24, 'RepairLevel': 'first'}, # Added based on common knowledge of IAF F-16 squadrons
    '119th': {'Type': 'F-16I Sufa', 'Starting_Strength': 24, 'RepairLevel': 'first'},
    '201st': {'Type': 'F-16I Sufa', 'Starting_Strength': 24, 'RepairLevel': 'first'},
    '253rd': {'Type': 'F-16I Sufa', 'Starting_Strength': 24, 'RepairLevel': 'first'},
    '120th': {'Type': 'KC-707 Saknayee', 'Starting_Strength': 8, 'RepairLevel': 'second'},
    '122nd': {'Type': 'Shavit', 'Starting_Strength': 3, 'RepairLevel': 'first'},
    '210th': {'Type': 'Eitan UAVs', 'Starting_Strength': 4, 'RepairLevel': 'first'},
}

# This lookup seems to be for squadron size *for certain calculations*, not initial strength.
# Renamed to reflect its likely purpose (e.g., for breakdown calculations or GCI fighter numbers).
# Ensure these numbers align with the rulebook's breakdown tables and GCI tables where squadron size is a factor.
SQUADRON_SIZE_FOR_CALCULATIONS = {
    '69th': 12, # F-15I typically operate in squadrons of 12-24, but often flights of 4-6
    '107th': 12, # F-16I squadrons are around 24, but often smaller units for specific calcs
    '109th F-16I Squadron': 12,
    '119th': 12,
    '201st': 12,
    '253rd': 12,
    '11 TFS': 10, # MiG-29 squadrons are 14, but breakdown table uses 10-plane column
    'Detachment 1': 4, # F-14 detachments are 4 planes
    '12 TFS': 10, # F-5E squadrons are 10
    '21 TFS': 8, # F-5E/F squadrons are 8
    '22 TFS': 8, # F-5E/F squadrons are 8
    '23 TFS': 10, # MiG-29 squadrons are 14, breakdown table uses 10-plane column
    'Detachment 2': 4,
    '32 TFS': 10, # F-4E squadrons are 10
    '33 TFS': 10, # F-4E squadrons are 10
    '41 TFS': 8, # F-5E squadrons are 8
    '42 TFS': 8,
    '43 TFS': 8,
    '51 TFS': 8, # F-7M squadrons are 8
    '52 TFS': 8,
    '53 TFS': 8,
    '71 TFS': 8,
    '81 TFS': 8, # F-14A squadrons are 8
    '82 TFS': 8,
    '83 TFS': 8,
    '61 TFS': 10,
    '62 TFS': 10,
    'Detachment 6': 4,
    '91 TFS': 7,
    '92 TFS': 7,
    '101 TFS': 5,
}

# Opinion Track Target Numbers (for rolling opinion dice)
OPINION_TARGET_NUMBERS = {
    "Ally": {"target_roll": 9, "value_range": (9, 10)},
    "Supporter": {"target_roll": 8, "value_range": (5, 8)},
    "Cordial": {"target_roll": 7, "value_range": (1, 4)},
    "Neutral": {"target_roll": 6, "value_range": (0, 0)},
    # For negative opinion, the logic for moving AGAINST the current value
    # would still use these targets based on the *absolute* value of the current opinion's category.
    # E.g., moving from Cordial with Iran (-2) towards Neutral (0) would use the 'Cordial' target number.
}


# Israeli Player Upgrades (from PersianIncursionRuleBook.pdf page 8 and rulebook page A1)
ISRAELI_UPGRADES = {
    "Third Arrow-2 battalion": {"cost": 35, "effect": "Allows a second Arrow battery to be deployed."},
    "Iron Dome expanded": {"cost": 40, "effect": "Reduces effectiveness of Iranian terror attacks by 20%."},
    "MALD (Miniature Air Launched Decoy)": {"cost": 30, "effect": "+2 on the Fighter Suppression Table."},
    "More aerial tankers": {"cost": 15, "unit": "aircraft", "effect": "Increases KC-707 fleet capacity."},
    "Jam-resistant GPS receivers (US-built ordnance only)": {"cost": 35, "effect": "Treat 1st Gen GPS Guidance as 2nd Gen for US ordnance."},
    "Jam-resistant GPS receivers (US-built and Israeli-built ordnance)": {"cost": 50, "effect": "Treat 1st Gen GPS Guidance as 2nd Gen for all ordnance."},
    "AIM-120D AMRAAM": {"cost": 30, "effect": "Range increase to 60 nm for Israeli F-16I/F-15I AIM-120s."},
    "EGBU-28C": {"cost": 40, "effect": "Can penetrate the Natanz centrifuge halls overhead protection."},
    "AGM-88 HARM Block 5": {"cost": 35, "effect": "Can lock onto and destroy GPS jammers with specific hit chance (8 on D10, 5 if jammer shut down)."},
}

# Iranian Player Upgrades (from PersianIncursionRuleBook.pdf page 7 and rulebook page A1)
IRANIAN_UPGRADES = {
    "Improved early warning radars": {"cost": 40, "effect": "+2 on the GCI Fighter Table."},
    "Improved air defense network": {"cost": 50, "effect": "-2 on Israeli Suter Attack Table."},
    "Bodyguard laser decoys/dazzlers (for Nuclear infrastructure)": {"cost": 10, "effect": "-4 for laser-guided ordnance on Nuclear PGM Attack Table."},
    "Bodyguard laser decoys/dazzlers (for Oil infrastructure)": {"cost": 25, "effect": "-4 for laser-guided ordnance on Oil PGM Attack Table."},
    "Bodyguard laser decoys/dazzlers (for Military targets)": {"cost": 45, "effect": "-4 for laser-guided ordnance on Military PGM Attack Table."},
    "GPS jammers (for Nuclear infrastructure)": {"cost": 30, "effect": "-3 or -1 modifier for GPS-guided ordnance on Nuclear PGM Attack Table."},
    "GPS jammers (for Nuclear and Oil infrastructure)": {"cost": 65, "effect": "-3 or -1 modifier for GPS-guided ordnance on Nuclear and Oil PGM Attack Table."},
    "High-fidelity decoys of an entire SAM battery": {"cost": 10, "unit": "battery", "effect": "Mimic visual, radar, and IR signatures. Not affected by Suter attacks."},
    "Pantsyr-S1E [SA-22] mobile gun/SAM system": {"cost": 15, "unit": "battery", "max_batteries": 2, "effect": "Adds mobile short-range gun/SAM system (max 2 batteries)."},
    "Additional Tor-M1 [SA-15] batteries": {"cost": 20, "unit": "battery", "effect": "Adds short-range SAMs (requires Russia Supporter)."},
    "S-300PMU-1 [SA-20] batteries": {"cost": 40, "unit": "battery", "effect": "Adds advanced long-range SAMs (requires Russia Supporter)."},
    "Buk-M1 [SA-11] batteries": {"cost": 25, "unit": "battery", "effect": "Adds advanced medium-range SAMs (requires Russia Supporter)."},
    "HQ-9 batteries": {"cost": 35, "unit": "battery", "effect": "Adds advanced long-range SAMs (requires PRC Supporter)."},
    "Sejil-2 ballistic missile battalion": {"cost": 30, "max_battalions": 2, "effect": "Can be launched the same turn as ordered (max 2 battalions)."},
    "R-27ER1 AAM upgrade for Iranian MiG-29": {"cost": 25, "effect": "Upgrades MiG-29 missiles (requires Russia Supporter)."},
    "PL-5E and PL-8 AAMs to replace AIM-9/Fatter from PRC": {"cost": 25, "effect": "Upgrades F-5E missiles (requires PRC Supporter)."},
    "EM-55 Guided propelled deepwater mines": {"cost": 35, "effect": "Allow mining of the Strait of Hormuz, +2 modifier to the Blockade Results Table die roll."},
}

# Extranational Reinforcements (from PersianIncursionRuleBook.pdf page 7)
IRANIAN_EXTRANATIONAL_REINFORCEMENTS = {
    "People's Republic of China": {
        "J-11 Flanker squadron": {"cost": 10, "type": "fighter", "quantity": 12, "effect": "Adds 12 J-11 aircraft to Iranian forces.", "requires_ally": True, "arrival_turns": 3},
        "J-10 squadron": {"cost": 10, "type": "fighter", "quantity": 12, "effect": "Adds 12 J-10 aircraft to Iranian forces.", "requires_ally": True, "arrival_turns": 3},
        "KJ-2000 Mainring AEW detachment": {"cost": 15, "type": "AEW", "effect": "+1 to Iranian GCI fighter rolls, +2 to PRC GCI fighter rolls.", "requires_ally": True, "arrival_turns": 3},
        "HQ-9 battery (Fixed Long-Range SAM)": {"cost": 8, "unit": "battery", "effect": "Adds HQ-9 SAM battery (will only defend oil infrastructure targets).", "requires_ally": True, "arrival_turns": 3},
    },
    "Russian Federation": {
        "MiG-29SMT Fulcrum squadron": {"cost": 10, "type": "fighter", "quantity": 12, "effect": "Adds 12 MiG-29SMT aircraft to Iranian forces.", "requires_ally": True, "arrival_turns": 3},
        "Su-27SM-1 Flanker squadron": {"cost": 10, "type": "fighter", "quantity": 12, "effect": "Adds 12 Su-27SM-1 aircraft to Iranian forces.", "requires_ally": True, "arrival_turns": 3},
        "MiG-31 Foxhound squadron": {"cost": 10, "type": "fighter", "quantity": 12, "effect": "Adds 12 MiG-31 aircraft to Iranian forces.", "requires_ally": True, "arrival_turns": 3},
        "A-50 Mainstay detachment": {"cost": 15, "type": "AEW", "effect": "+1 to Iranian GCI fighter rolls, +2 to Russian GCI fighter rolls.", "requires_ally": True, "arrival_turns": 3},
        "Tor-M1 battery": {"cost": 6, "unit": "battery", "effect": "Adds Tor-M1 SAM battery.", "requires_ally": True, "arrival_turns": 3},
        "S-300PMU-1 [SA-20] battery": {"cost": 10, "unit": "battery", "effect": "Adds S-300PMU-1 SAM battery.", "requires_ally": True, "arrival_turns": 3},
        "Buk-M1 [SA-11] battery": {"cost": 8, "unit": "battery", "effect": "Adds Buk-M1 SAM battery.", "requires_ally": True, "arrival_turns": 3},
    }
}

# === OPINION → INCOME TABLES (verbatim from the book) ===
# Ranges are inclusive; value is the *track value* (−10 … +10).
DOMESTIC_OPINION_INCOME = {
    "israel": [
        (9,  10, (6, 7, 10)),  # +9 or more  -> 6 PP, 7 IP, 10 MP
        (5,   8, (5, 6, 10)),  # +5..+8
        (2,   4, (4, 5, 10)),  # +4..+2
        (-1,  1, (3, 5,  9)),  # +1..-1
        (-4, -2, (2, 3,  8)),  # -2..-4
        (-8, -5, (1, 1,  8)),  # -5..-8
        (-10,-9, (0, 0,  6)),  # -9 or less
    ],
    "iran": [
        (9,  10, (1, 0, 0)),   # +9 or more  (pro-Israel) -> 1 PP, 0 IP, 0 MP
        (5,   8, (2, 1, 1)),   # +5..+8
        (2,   4, (3, 2, 3)),   # +4..+2
        (-1,  1, (4, 3, 5)),   # +1..-1
        (-4, -2, (5, 4, 6)),   # -2..-4
        (-8, -5, (6, 5, 6)),   # -5..-8
        (-10,-9, (7, 6, 6)),   # -9 or less (pro-Iran)
    ],
}

# Third-party table: encode *exactly what the book gives*. Each entry is:
# (min, max, {"israel": (PP,IP,MP)} or {"iran": (PP,IP,MP)})
# Below I’ve filled **USA** precisely from your scan; use the same pattern
# to add PRC, Russia, Saudi/GCC, UN, Jordan, Turkey.
THIRD_PARTY_OPINION_INCOME = {
    "usa": [
        (9, 10, {"israel": (4, 4, 4)}),
        (5,  8, {"israel": (3, 3, 2)}),
        (1,  4, {"israel": (1, 2, 1)}),
        (0,  0, {"israel": (0, 1, 0)}),
        (-4,-1, {"iran":   (1, 0, 0)}),
        (-8,-5, {"iran":   (2, 1, 1)}),
        (-10,-9,{"iran":   (4, 2, 1)}),
    ],
    # --- Data transcribed from the table image ---
    "china": [
        (9, 10, {"israel": (2, 2, 0)}),
        (5,  8, {"israel": (1, 2, 0)}),
        (1,  4, {"israel": (1, 0, 0)}),
        (0,  0, {"iran":   (1, 0, 0)}),
        (-4,-1, {"iran":   (1, 1, 2)}),
        (-8,-5, {"iran":   (2, 2, 2)}),
        (-10,-9,{"iran":   (4, 4, 4)}),
    ],
    "russia": [
        (9, 10, {"israel": (2, 2, 0)}),
        (5,  8, {"israel": (1, 2, 0)}),
        (1,  4, {"israel": (1, 0, 0)}),
        (0,  0, {"iran":   (1, 0, 0)}),
        (-4,-1, {"iran":   (1, 2, 0)}),
        (-8,-5, {"iran":   (2, 3, 2)}),
        (-10,-9,{"iran":   (4, 4, 4)}),
    ],
    "sa": [
        (9, 10, {"israel": (3, 2, 2)}),
        (5,  8, {"israel": (2, 2, 0)}),
        (1,  4, {"israel": (1, 0, 0)}),
        (0,  0, {"israel": (0, 0, 0)}), # No income for Neutral
        (-4,-1, {"iran":   (1, 0, 0)}),
        (-8,-5, {"iran":   (2, 0, 0)}),
        (-10,-9,{"iran":   (3, 2, 2)}),
    ],
    "un": [
        (9, 10, {"israel": (4, 1, 0)}),
        (5,  8, {"israel": (2, 0, 0)}),
        (1,  4, {"israel": (1, 0, 0)}),
        (0,  0, {"israel": (0, 0, 0)}), # No income for Neutral
        (-4,-1, {"iran":   (1, 0, 0)}),
        (-8,-5, {"iran":   (2, 0, 0)}),
        (-10,-9,{"iran":   (4, 1, 0)}),
    ],
    "jordan": [
        (9, 10, {"israel": (0, 0, 0)}), # -- in table
        (5,  8, {"israel": (2, 0, 0)}),
        (1,  4, {"israel": (1, 0, 0)}),
        (0,  0, {"israel": (0, 0, 0)}), # No income for Neutral
        (-4,-1, {"iran":   (1, 0, 0)}),
        (-8,-5, {"iran":   (2, 0, 0)}),
        (-10,-9,{"iran":   (3, 2, 2)}),
    ],
    "turkey": [
        (9, 10, {"israel": (3, 2, 2)}),
        (5,  8, {"israel": (2, 1, 0)}),
        (1,  4, {"israel": (1, 0, 0)}),
        (0,  0, {"israel": (0, 0, 0)}), # No income for Neutral
        (-4,-1, {"iran":   (1, 0, 0)}),
        (-8,-5, {"iran":   (2, 1, 0)}),
        (-10,-9,{"iran":   (3, 2, 0)}),
    ],
}
# ======== rules_blob.py APPEND: canonical parse + RULES ========
import re

def _parse_cost_to_map(cost_str: str):
    """
    Convert Cost like '3P, 2I, 4M' or '--' into {'pp':3,'ip':2,'mp':4}.
    Matches engine expectations (lowercase keys).  (see game_engine.get_legal_actions) 
    """
    m = {"pp":0,"ip":0,"mp":0}
    s = (cost_str or "").strip()
    if not s or s == "--":
        return m
    for amt, unit in re.findall(r'(\d+)\s*([PIM])', s.upper()):
        m[{"P":"pp","I":"ip","M":"mp"}[unit]] += int(amt)
    return m

def _extract_flags(notes: str):
    flags = set()
    s = (notes or "").lower()
    for k in ("dirty","covert","backfire","requires"):
        if k in s: flags.add(k)
    return sorted(flags)

def _parse_requires(notes: str):
    """
    Very light extractor for common 'Requires:' phrases used in your card lists.
    Keeps the *exact* string (no heuristic logic) so the engine can check it.
    """
    if not notes: return {}
    m = re.search(r"requires[:\s]+(.+)$", notes, re.IGNORECASE)
    return {"text": m.group(1).strip()} if m else {}

def _structure_cards(card_list):
    out = {}
    for c in card_list:
        cid = int(c["No"])
        out[cid] = {
            "id": cid,
            "name": c.get("Name",""),
            "qty": int(c.get("Quantity",1)),
            "cost_map": _parse_cost_to_map(c.get("Cost","")),
            "effect_text": c.get("Effect",""),
            "notes_text": c.get("Notes",""),
            "flags": _extract_flags(c.get("Notes","")),
            "requires": _parse_requires(c.get("Notes","")),
        }
    return out

# ---- Build structured card maps from your existing lists (keeps originals) ----
CARDS_STRUCTURED = {
    "iran":   _structure_cards(IRAN_CARDS),
    "israel": _structure_cards(ISRAEL_CARDS),
}

# ---- Canonical rule toggles/knobs used by engine & env (follow rulebook) ----
RIVER_RULES = {
    "slots": 7,            # river size (matches engine bootstrap) 
    "discard_rightmost": True,   # end-of-phase/day discard policy
}

RESTRIKE_RULES = {
    "plan_delay_turns": 1,    # plan this turn -> earliest execute next eligible window
    "execute_window_turns": 1 # strict D+1 execution window before it goes stale
}

# Action costs (kept conservative; engine can read these if you wire costs there)
ACTION_COSTS = {
    "PLAN_STRIKE":   {"mp":3, "ip":3},
    "SPECWAR":       {"mp_min":1, "mp_max":3, "ip_min":1, "ip_max":4},
    "IRBM_LAUNCH":   {"mp_per_battalion":1, "max_battalions":4},
    "HORMUZ_TRY":    {"mp_min":1, "mp_max":7, "pp_max":2}
}

# Airspace access by opinion thresholds (exact numbers can be tuned to book)
AIRSPACE_RULES = {
    # corridor -> {country key, min opinion needed}
    "north":   {"country": "turkey", "min_op": 1},
    "central": {"country": "jordan", "min_op": 1},
    "south":   {"country": "sa",     "min_op": 1},   # SA/GCC track
}

# Victory thresholds (example: refine to match your PDF tables)
VICTORY_THRESHOLDS = {
    "tactical": {"refinery_output_pct": 60, "natanz_destroyed": False},
    "decisive": {"refinery_output_pct": 40, "natanz_destroyed": True}
}

# ---- Opinion income tables (if not already provided) ----
# If you already export DOMESTIC_OPINION_INCOME/THIRD_PARTY_OPINION_INCOME above,
# these assignments simply alias them into canonical names used by the engine.
OPINION_INCOME_TABLE = globals().get("DOMESTIC_OPINION_INCOME", {})
THIRD_PARTY_INCOME_TABLE = globals().get("THIRD_PARTY_OPINION_INCOME", {})

# ---- Normalize squadron/OOB tables into one place the engine can read quickly ----
SQUADRONS = {
    "israel": globals().get("ISRAELI_SQUADRONS_SETUP", {}),
    "iran":   globals().get("IRANIAN_SQUADRONS_SETUP", {}),
}
SQUADRON_SIZE_FOR_CALCULATIONS = globals().get("SQUADRON_SIZE_FOR_CALCULATIONS", {})

# ---- Build canonical RULES (single dict), preserving original exports ----
RULES = {
    # core data
    "IRAN_CARDS": IRAN_CARDS,
    "ISRAEL_CARDS": ISRAEL_CARDS,
    "IRAN_CARD_MAP": IRAN_CARD_MAP,
    "ISRAEL_CARD_MAP": ISRAEL_CARD_MAP,
    "cards_structured": CARDS_STRUCTURED,
    "SAM_COMBAT_TABLE": SAM_COMBAT_TABLE,
    "AAA_COMBAT_TABLE": globals().get("AAA_COMBAT_TABLE", {}),
    "PGM_ATTACK_TABLE": globals().get("PGM_ATTACK_TABLE", {}),
    "TARGET_DEFENSES": globals().get("TARGET_DEFENSES", {}),
    "VALID_TARGETS": globals().get("VALID_TARGETS", {}),
    "OPINION_TARGET_NUMBERS": globals().get("OPINION_TARGET_NUMBERS", {}),
    "SQUADRONS": SQUADRONS,
    "SQUADRON_SIZE_FOR_CALCULATIONS": SQUADRON_SIZE_FOR_CALCULATIONS,

    # rules knobs
    "RIVER_RULES": RIVER_RULES,
    "RESTRIKE_RULES": RESTRIKE_RULES,
    "ACTION_COSTS": ACTION_COSTS,
    "AIRSPACE_RULES": AIRSPACE_RULES,
    "VICTORY_THRESHOLDS": VICTORY_THRESHOLDS,

    # economy/opinion
    "OPINION_INCOME_TABLE": OPINION_INCOME_TABLE,
    "THIRD_PARTY_INCOME_TABLE": THIRD_PARTY_INCOME_TABLE,
}
