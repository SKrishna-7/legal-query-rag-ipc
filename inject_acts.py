import json

new_sections = [
    {
        "section_number": "BNS 61",
        "title": "Criminal Conspiracy",
        "chapter": "V",
        "chapter_title": "Of Criminal Conspiracy",
        "full_text": "When two or more persons agree to do, or cause to be done, an illegal act, or an act which is not illegal by illegal means, such an agreement is designated a criminal conspiracy.",
        "punishment": "Punished in the same manner as if he had abetted such offence",
        "essential_ingredients": [
          "Agreement between two or more persons",
          "The agreement is to do an illegal act, or a legal act by illegal means"
        ],
        "related_sections": [],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Court of Session",
        "compoundable": False,
        "keywords": ["conspiracy", "agreement", "illegal act"],
        "mens_rea_required": True,
        "actus_reus": "Entering into an agreement for an illegal act",
        "maximum_punishment": "Depends on the offence",
        "minimum_punishment": "Depends on the offence"
    },
    {
        "section_number": "PC 7",
        "title": "Offence relating to public servant being bribed",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "Any public servant who obtains or accepts or attempts to obtain from any person, an undue advantage, with the intention to perform or cause performance of public duty improperly or dishonestly.",
        "punishment": "Imprisonment for 3 to 7 years and fine",
        "essential_ingredients": [
          "The accused is a public servant",
          "Obtains, accepts, or attempts to obtain an undue advantage",
          "Intention to perform public duty improperly or dishonestly"
        ],
        "related_sections": ["PC 7A", "PC 8"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["public servant", "bribe", "undue advantage", "corruption"],
        "mens_rea_required": True,
        "actus_reus": "Accepting or obtaining an undue advantage",
        "maximum_punishment": "7 years",
        "minimum_punishment": "3 years"
    },
    {
        "section_number": "PC 7A",
        "title": "Taking undue advantage to influence public servant",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "Whoever accepts or obtains any undue advantage as a motive or reward to induce a public servant, by corrupt or illegal means or by exercise of his personal influence to perform a public duty improperly.",
        "punishment": "Imprisonment for 3 to 7 years and fine",
        "essential_ingredients": [
          "Accepts or obtains an undue advantage",
          "As a motive or reward to induce a public servant",
          "By corrupt or illegal means or personal influence"
        ],
        "related_sections": ["PC 7", "PC 8"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["middleman", "influence", "undue advantage", "induce"],
        "mens_rea_required": True,
        "actus_reus": "Accepting undue advantage to influence public servant",
        "maximum_punishment": "7 years",
        "minimum_punishment": "3 years"
    },
    {
        "section_number": "PC 8",
        "title": "Offence relating to bribing of a public servant",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "Any person who gives or promises to give an undue advantage to another person or persons, with intention to induce a public servant to perform improperly a public duty.",
        "punishment": "Imprisonment up to 7 years or fine or both",
        "essential_ingredients": [
          "Gives or promises to give an undue advantage",
          "Intention to induce a public servant to perform public duty improperly"
        ],
        "related_sections": ["PC 7"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["bribing", "give bribe", "induce", "undue advantage"],
        "mens_rea_required": True,
        "actus_reus": "Giving or promising an undue advantage",
        "maximum_punishment": "7 years",
        "minimum_punishment": "None"
    },
    {
        "section_number": "PC 9",
        "title": "Offence relating to bribing a public servant by a commercial organisation",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "A commercial organisation shall be guilty of an offence if any person associated with it gives or promises to give any undue advantage to a public servant intending to obtain or retain business or an advantage in the conduct of business.",
        "punishment": "Fine",
        "essential_ingredients": [
          "A person associated with a commercial organisation gives or promises an undue advantage",
          "Intention to obtain or retain business or advantage for the commercial organisation",
          "The recipient is a public servant"
        ],
        "related_sections": ["PC 10"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["commercial organisation", "bribe", "retain business"],
        "mens_rea_required": True,
        "actus_reus": "Giving an undue advantage on behalf of an organisation",
        "maximum_punishment": "Fine",
        "minimum_punishment": "Fine"
    },
    {
        "section_number": "PC 10",
        "title": "Person in charge of commercial organisation to be guilty",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "Where an offence under section 9 is committed by a commercial organisation, and such offence is proved to have been committed with the consent or connivance of any director, manager, secretary or other officer shall be guilty of the offence.",
        "punishment": "Imprisonment for 3 to 7 years and fine",
        "essential_ingredients": [
          "Offence under section 9 committed by a commercial organisation",
          "Committed with consent or connivance of director, manager, or officer"
        ],
        "related_sections": ["PC 9"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["director", "consent", "connivance", "commercial organisation"],
        "mens_rea_required": True,
        "actus_reus": "Consenting or conniving to an offence under section 9",
        "maximum_punishment": "7 years",
        "minimum_punishment": "3 years"
    },
    {
        "section_number": "PC 11",
        "title": "Public servant obtaining undue advantage, without consideration from person concerned in proceeding",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "Whoever, being a public servant, accepts or obtains or attempts to obtain for himself, or for any other person, any undue advantage without consideration, or for a consideration which he knows to be inadequate, from any person whom he knows to have been, or to be, or to be likely to be concerned in any proceeding or business transacted by such public servant.",
        "punishment": "Imprisonment for 6 months to 5 years and fine",
        "essential_ingredients": [
          "The accused is a public servant",
          "Accepts or obtains undue advantage without consideration or inadequate consideration",
          "From a person concerned in a proceeding or business transacted by the public servant"
        ],
        "related_sections": ["PC 7"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["without consideration", "inadequate consideration", "proceeding"],
        "mens_rea_required": True,
        "actus_reus": "Obtaining undue advantage without adequate consideration",
        "maximum_punishment": "5 years",
        "minimum_punishment": "6 months"
    },
    {
        "section_number": "PC 12",
        "title": "Punishment for abetment of offences",
        "chapter": "III",
        "chapter_title": "Offences and Penalties",
        "full_text": "Whoever abets any offence punishable under this Act, whether or not that offence is committed in consequence of that abetment, shall be punishable with imprisonment for a term which shall be not less than three years but which may extend to seven years and shall also be liable to fine.",
        "punishment": "Imprisonment for 3 to 7 years and fine",
        "essential_ingredients": [
          "Abets an offence punishable under the PC Act"
        ],
        "related_sections": ["PC 7", "PC 11"],
        "cognizable": True,
        "bailable": False,
        "triable_by": "Special Judge",
        "compoundable": False,
        "keywords": ["abetment", "abet", "assistance"],
        "mens_rea_required": True,
        "actus_reus": "Abetting a PC Act offence",
        "maximum_punishment": "7 years",
        "minimum_punishment": "3 years"
    }
]

kb_path = "data/processed/ipc_sections/ipc_complete.json"
try:
    with open(kb_path, "r") as f:
        data = json.load(f)
except Exception as e:
    data = []

# Remove existing to prevent duplicates if run multiple times
data = [d for d in data if d.get("section_number") not in [s["section_number"] for s in new_sections]]

# Append new
data.extend(new_sections)

with open(kb_path, "w") as f:
    json.dump(data, f, indent=2)

print("Injected PC Act and BNS sections into the knowledge base!")
