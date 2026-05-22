"""Run this once in WSL2: python patch_questions.py"""
import re, pathlib

NEW_QA = '''# 10 clusters x 5 questions = 50 total.
# Every question repeats ONE dominant keyword so embeddings are near-identical
# -> retriever picks k=1 (same chunk) -> same doc hash -> cache hit.
BENCHMARK_QA = [
    # C1: Quirrell
    {"question": "Who was Quirrell and what secret was he hiding?",
     "ground_truth": "Professor Quirrell was the Defence Against the Dark Arts teacher secretly possessed by Voldemort, who lived on the back of Quirrell's head."},
    {"question": "Why was Quirrell secretly working for Voldemort?",
     "ground_truth": "Quirrell was possessed by Voldemort and directed to obtain the Sorcerer's Stone so Voldemort could return to human form."},
    {"question": "What was Quirrell's true motive for wanting the Sorcerer's Stone?",
     "ground_truth": "Quirrell, under Voldemort's control, wanted the Sorcerer's Stone to restore Voldemort to a human body."},
    {"question": "How did Harry overpower Quirrell in the underground chamber?",
     "ground_truth": "Harry's love-based magical protection caused Quirrell to burn at Harry's touch, since Quirrell was possessed by Voldemort."},
    {"question": "What happened to Quirrell after Harry grabbed his face?",
     "ground_truth": "Quirrell burned and crumbled at Harry's touch because of Harry's love-based magical protection against Voldemort, and Voldemort fled Quirrell's dying body."},

    # C2: Sorcerer's Stone
    {"question": "What is the Sorcerer's Stone?",
     "ground_truth": "The Sorcerer's Stone is a magical item that can grant its owner immortality."},
    {"question": "What power does the Sorcerer's Stone grant its owner?",
     "ground_truth": "The Sorcerer's Stone grants its owner immortality."},
    {"question": "Why did Voldemort want the Sorcerer's Stone?",
     "ground_truth": "Voldemort wanted the Sorcerer's Stone to return to human form and regain immortality."},
    {"question": "What did Dumbledore do with the Sorcerer's Stone after Harry defeated Quirrell?",
     "ground_truth": "Dumbledore destroyed the Sorcerer's Stone so it could never be used."},
    {"question": "What did Voldemort promise Harry in exchange for the Sorcerer's Stone?",
     "ground_truth": "Voldemort promised to bring Harry's parents back from the dead if Harry gave him the Sorcerer's Stone."},

    # C3: Basilisk
    {"question": "What is the Basilisk in the Chamber of Secrets?",
     "ground_truth": "The Basilisk is a giant snake that lived in the Chamber of Secrets, controlled by the Heir of Slytherin. It petrified students and killed with its gaze."},
    {"question": "How did the Basilisk attack students at Hogwarts?",
     "ground_truth": "The Basilisk killed with its direct gaze. Students who saw only its reflection or through another medium were petrified rather than killed."},
    {"question": "How did Harry kill the Basilisk in the Chamber of Secrets?",
     "ground_truth": "Harry used the Sword of Godric Gryffindor to impale the Basilisk in the roof of the mouth, killing it."},
    {"question": "What injury did Harry suffer when he killed the Basilisk?",
     "ground_truth": "A Basilisk fang pierced Harry's arm, poisoning him with venom that would have killed him."},
    {"question": "How did Fawkes help Harry fight the Basilisk in the Chamber of Secrets?",
     "ground_truth": "Fawkes attacked the Basilisk's eyes, blinding it so its gaze could no longer kill, then brought Harry the Sorting Hat from which appeared the Sword of Godric Gryffindor."},

    # C4: Fawkes
    {"question": "Who is Fawkes and what role did Fawkes play in the Chamber of Secrets?",
     "ground_truth": "Fawkes is Dumbledore's phoenix. In the Chamber of Secrets Fawkes blinded the Basilisk, delivered the Sorting Hat to Harry, and healed Harry's Basilisk fang wound with phoenix tears."},
    {"question": "How did Fawkes heal Harry's Basilisk fang wound in the Chamber of Secrets?",
     "ground_truth": "Fawkes cried on Harry's Basilisk fang wound. Phoenix tears have healing powers that neutralised the venom and saved Harry's life."},
    {"question": "What did Fawkes deliver to Harry inside the Chamber of Secrets?",
     "ground_truth": "Fawkes delivered the Sorting Hat to Harry, from which the Sword of Godric Gryffindor appeared."},
    {"question": "Why were Fawkes's tears able to save Harry from the Basilisk venom?",
     "ground_truth": "Phoenix tears have healing powers. Fawkes cried on Harry's Basilisk fang wound and the tears neutralised the venom."},
    {"question": "How did Fawkes blind the Basilisk during Harry's fight in the Chamber of Secrets?",
     "ground_truth": "Fawkes attacked the Basilisk's eyes directly, blinding it so it could no longer kill with its gaze."},

    # C5: Pettigrew / Scabbers
    {"question": "Who was Peter Pettigrew hiding as and for how long?",
     "ground_truth": "Peter Pettigrew was hiding as Ron Weasley's pet rat Scabbers for twelve years to avoid capture for betraying the Potters to Voldemort."},
    {"question": "What was Peter Pettigrew's Animagus animal form?",
     "ground_truth": "Peter Pettigrew's Animagus form was a rat. He had been hiding as Ron's pet rat Scabbers."},
    {"question": "What crime did Peter Pettigrew commit that was blamed on Sirius Black?",
     "ground_truth": "Pettigrew betrayed the Potters' location to Voldemort, then faked his own death to frame Sirius Black for the crime."},
    {"question": "Why did Peter Pettigrew hide as the rat Scabbers for twelve years?",
     "ground_truth": "Pettigrew hid as Scabbers to avoid being caught for betraying the Potters to Voldemort and framing Sirius Black."},
    {"question": "How was Peter Pettigrew exposed as the real traitor who had been hiding as Scabbers?",
     "ground_truth": "Lupin and Sirius Black forced Pettigrew back into his human form in the Shrieking Shack, revealing him as an Animagus who had been hiding as Ron's rat Scabbers."},

    # C6: Sirius Black
    {"question": "Who is Sirius Black and what is his relationship to Harry Potter?",
     "ground_truth": "Sirius Black is Harry Potter's godfather and was his parents' best friend."},
    {"question": "Why was Sirius Black sent to Azkaban prison?",
     "ground_truth": "Sirius Black was wrongly convicted of betraying the Potters to Voldemort and murdering Peter Pettigrew. The real culprit was Pettigrew, who framed Sirius."},
    {"question": "How was Sirius Black proved innocent of betraying the Potters?",
     "ground_truth": "Lupin and Sirius forced Peter Pettigrew back into human form in the Shrieking Shack, revealing that Pettigrew, not Sirius, had betrayed the Potters."},
    {"question": "How did Sirius Black escape after Pettigrew fled and the Dementors attacked?",
     "ground_truth": "Harry and Hermione used the Time-Turner to travel back in time. They freed Buckbeak the Hippogriff and Sirius escaped by flying away on Buckbeak."},
    {"question": "What is Sirius Black's connection to the flying motorcycle mentioned in the series?",
     "ground_truth": "The flying motorcycle used to deliver baby Harry to the Dursleys belonged to Sirius Black and was borrowed by Hagrid."},

    # C7: Priori Incantatem
    {"question": "What is Priori Incantatem?",
     "ground_truth": "Priori Incantatem is a magical connection that occurs when two wands sharing the same core are forced to duel, causing one wand to echo spells previously cast by the other."},
    {"question": "What causes Priori Incantatem to happen between two wands?",
     "ground_truth": "Priori Incantatem occurs when two wands that share the same core are forced into combat with each other."},
    {"question": "What happened during Priori Incantatem between Harry's and Voldemort's wands in the graveyard?",
     "ground_truth": "Harry's wand forced Voldemort's wand to disgorge the spirits of people Voldemort had most recently killed, including Harry's parents and Cedric Diggory."},
    {"question": "Which spirits appeared from Voldemort's wand during Priori Incantatem?",
     "ground_truth": "The spirits of Harry's parents and Cedric Diggory appeared from Voldemort's wand during Priori Incantatem and briefly shielded Harry."},
    {"question": "How did Priori Incantatem help Harry escape from Voldemort in the graveyard?",
     "ground_truth": "The spirit echoes from Priori Incantatem shielded Harry while he broke the wand connection, summoned the Triwizard Cup Portkey, and escaped with Cedric's body."},

    # C8: Time-Turner
    {"question": "What is a Time-Turner and who used one in Prisoner of Azkaban?",
     "ground_truth": "A Time-Turner is a magical device that allows the user to travel back in time. Hermione used one all year to attend multiple classes simultaneously."},
    {"question": "How did Hermione use the Time-Turner throughout Prisoner of Azkaban?",
     "ground_truth": "Hermione used the Time-Turner to travel back in time so she could attend multiple classes scheduled at the same time."},
    {"question": "What did Harry and Hermione achieve by using the Time-Turner to travel back three hours?",
     "ground_truth": "Harry and Hermione used the Time-Turner to travel back three hours, freeing Buckbeak the Hippogriff from execution and rescuing Sirius Black, who escaped on Buckbeak."},
    {"question": "How did Harry realise he was the one who cast the stag Patronus after using the Time-Turner?",
     "ground_truth": "After travelling back in time with the Time-Turner, Harry watched events replay and realised the distant figure he had seen cast the Patronus was himself, so he cast it again to scatter the Dementors."},
    {"question": "Why did Harry and Hermione need the Time-Turner to save both Sirius and Buckbeak?",
     "ground_truth": "Sirius had been condemned to the Dementor's Kiss and Buckbeak to execution. The only way to save both without being seen was to travel back in time using the Time-Turner."},

    # C9: Horcrux
    {"question": "What is a Horcrux?",
     "ground_truth": "A Horcrux is an object in which a dark wizard hides a fragment of their soul, granting immortality as long as the Horcrux survives."},
    {"question": "How does a Horcrux grant its creator immortality?",
     "ground_truth": "A Horcrux safeguards a portion of the creator's soul outside their body. As long as the Horcrux exists, the creator cannot truly die."},
    {"question": "How many Horcruxes did Voldemort create?",
     "ground_truth": "Voldemort created seven Horcruxes."},
    {"question": "Which of Voldemort's Horcruxes were destroyed before Deathly Hallows?",
     "ground_truth": "Two Horcruxes were destroyed before Deathly Hallows: Tom Riddle's diary and Voldemort's grandfather's ring."},
    {"question": "How did Harry accidentally become one of Voldemort's Horcruxes?",
     "ground_truth": "When Voldemort's Killing Curse rebounded off baby Harry, a fragment of Voldemort's soul lodged inside Harry, inadvertently making Harry an unintended Horcrux."},

    # C10: Deathly Hallows
    {"question": "What are the Deathly Hallows?",
     "ground_truth": "The Deathly Hallows are three sacred magical objects: the Elder Wand, the Resurrection Stone, and the Invisibility Cloak."},
    {"question": "What are the three objects that make up the Deathly Hallows?",
     "ground_truth": "The three Deathly Hallows are the Elder Wand (an unbeatable wand), the Resurrection Stone (which revives the dead), and the Invisibility Cloak (which is infallible)."},
    {"question": "What does the Elder Wand do and why did Voldemort seek it among the Deathly Hallows?",
     "ground_truth": "The Elder Wand is an unbeatable wand. Voldemort sought it among the Deathly Hallows because he believed it would make him invincible."},
    {"question": "What does the Resurrection Stone do in the Deathly Hallows?",
     "ground_truth": "The Resurrection Stone has the power to bring others back from the dead."},
    {"question": "Which of the Deathly Hallows was Voldemort pursuing and why?",
     "ground_truth": "Voldemort was pursuing the Elder Wand because it is an unbeatable wand that he believed would make him invincible."},
]
'''

for path in ["main.py", "run_s6.py"]:
    p = pathlib.Path(path)
    src = p.read_text(encoding="utf-8")
    # replace from the comment above BENCHMARK_QA to the closing ]
    new_src = re.sub(
        r"(#[^\n]*\n)*BENCHMARK_QA = \[.*?\]\n",
        NEW_QA,
        src,
        count=1,
        flags=re.DOTALL,
    )
    if new_src == src:
        print(f"WARNING: pattern not found in {path}")
    else:
        p.write_text(new_src, encoding="utf-8")
        n = NEW_QA.count('"question"')
        print(f"Patched {path} - {n} questions")
