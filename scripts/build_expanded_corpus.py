"""400 new synthetic regression inputs; no output-quality labels or model answers.

160 individually authored inputs, 200 structured combinations, 40 paragraphs.
The fixed corpus is saved before inference, without selecting cases by outputs.
"""
import json
from pathlib import Path
from accuracy_corpus import CASES as OLD_CASES

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {
    'new_daily': '''What is your name?
Have you eaten?
Kashmir belongs to Kashmiris.
Please tell me your full name.
I left your lunch in the small cupboard.
The milk boiled over while I was answering the phone.
We have enough rice for everyone except the unexpected guests.
My grandfather cannot find his walking stick.
The neighbour returned the bowl without its lid.
Someone has moved the shoes from the doorway.
The baby woke up as soon as I turned off the light.
There is a loose button on your coat.
Please keep the soup warm until she returns.
I washed these clothes yesterday, but they are still wet.
The tap has been dripping since Monday.
You can borrow my umbrella if you bring it back.''',
    'new_scope': '''Only my youngest cousin has seen this photograph.
My youngest cousin has only seen this photograph.
I never said you should stop learning.
I said you should never stop learning.
Not all of the windows were closed.
All of the windows were not closed.
Nobody except the driver knew the way.
Everyone but me remembered the appointment.
She neither borrowed the book nor bought it.
He cannot possibly have finished already.
I am not saying that your explanation is wrong.
You need not answer every question immediately.
Please leave no more than three chairs outside.
At least one of these keys should fit the lock.
Unless you call first, they may not be at home.
Even if he apologizes, I will still need some time.''',
    'new_time': '''I had been working here for six years when the shop closed.
She will have arrived before we finish cooking.
We were about to leave when the power went out.
Tomorrow at this time, I will be travelling to Srinagar.
I would have waited if you had told me you were coming.
If we had taken the earlier bus, we would be home now.
She has not visited us since her daughter was born.
He used to sing while he repaired the bicycles.
The meeting was postponed twice before it finally took place.
By the end of winter, they had run out of firewood.
I am getting used to waking up before sunrise.
We will keep looking until we find the missing receipt.
The work should have been completed last Friday.
She might have left the parcel at the wrong house.
It has been raining on and off all afternoon.
I have already told him what you told me yesterday.''',
    'new_people': '''Amina asked Zoya to bring her notebook to school.
Zoya asked Amina whether the notebook was hers.
My mother's older sister lives with her youngest son.
His daughter gave my son the toy that her brother had lost.
The teacher thanked the students who had stayed behind.
The students thanked the teacher who had helped them.
Both brothers blamed themselves for the misunderstanding.
Each of the sisters brought her own lunch.
Their parents are visiting ours next weekend.
Tell your uncle that my aunt has called twice.
The woman beside the bus stop is my father's cousin.
The boy whose bicycle was stolen is waiting outside.
Whoever arrives first should open the gate.
Those who cannot attend may send a written message.
I met the person you said you wanted me to meet.
She introduced herself before asking us our names.''',
    'new_emotions': '''I am too upset to explain everything right now.
You made me laugh even though I was having a terrible day.
I feel embarrassed about losing my temper yesterday.
It bothers me when you interrupt before I finish speaking.
I miss the way we used to spend our evenings together.
Please do not mistake my silence for agreement.
I am relieved that you reached home safely.
She was disappointed, but she did not blame anyone.
I cannot believe you remembered such a small detail.
He looked cheerful, although he had barely slept.
There is no point in arguing about something we cannot change.
I was joking, but I understand why it hurt you.
Let us talk about this after we have both calmed down.
You do not need to pretend that everything is fine.
I appreciate your help more than I know how to say.
I would rather hear an honest answer than a comforting lie.''',
    'new_numbers': '''The total is 1,250 rupees, including delivery.
Please arrive at 7:45 a.m., not 7:45 p.m.
The temperature fell from 12 degrees to minus 3 overnight.
Two and a half kilograms of flour should be enough.
The first bus leaves at 06:05 and the last at 21:30.
My room number is 204, on the second floor.
We sold 18 of the 25 tickets and gave away two.
The receipt says 850, but I paid 800.
The distance is approximately 3.5 kilometres.
She scored 87 out of 100 on the final examination.
The shop is closed from 1 to 2 every afternoon.
Write the date as 18 October 2026.
There are twelve bottles in each box and seven boxes in the van.
Half of the money is yours; the other half belongs to your sister.
The price increased by ten percent, then decreased by five percent.
Please call extension 116 if nobody answers the main number.''',
    'new_places': '''My family moved from Kashmir to Delhi many years ago.
We travelled from Jammu to Srinagar through the mountains.
The old bridge connects the village to the main road.
Kashmiri is spoken in our home, but English is used at work.
She writes the names of villages in both scripts.
The boat was tied to a wooden post beside the lake.
Fresh snow covered the roofs in Gulmarg this morning.
The road to Pahalgam was blocked by fallen rocks.
He bought saffron for his grandmother in Pampore.
Our relatives in Baramulla sent us a basket of apples.
The library is opposite the mosque, behind the bakery.
Keep walking until the road divides near the walnut tree.
We took the narrow path along the river instead of the highway.
The address on the envelope is no longer correct.
People from several villages came to repair the damaged bridge.
I want my children to learn the language their grandparents speak.''',
    'new_informal': '''whats your name again i forgot sorry
im hungry but i dont want to eat alone
can u bring the bag i left near ur door
no no thats not what i meant at all
wait... are you actually leaving today?
WHY did nobody tell me the meeting was cancelled?
okay, fine, let's start again.
I'm not angry; I'm just really, really tired.
she said "come tomorrow" but tomorrow is a holiday
I can't find it anywhere!!!
hello :) is anyone there?
Thanks! That helped a lot 😊
Please send the file named notes_v2.txt.
My email is example@example.com.
The sign reads: "No entry after sunset."
First wash your hands; then help me set the table.''',
    'new_requests': '''Could you lower your voice while the baby is sleeping?
Would you mind checking whether the back door is locked?
Tell me honestly if you think I have misunderstood.
Please remind me to bring the documents tomorrow.
Can you explain the difference using a simpler example?
Which of these two buses stops near the university?
How long have you known the people who live upstairs?
Why was the parcel returned without being delivered?
Who is responsible for turning off the lights at night?
Do you know whether the shop accepts card payments?
May I leave my suitcase here for a few minutes?
What would you do if you were in my position?
Is there another way to reach the station before noon?
Could we move the appointment to a later time?
Please read the last sentence aloud once more.
Do you want me to wait here or come with you?''',
    'new_contrasts': '''The bag is light, but the box is heavy.
The room is bright even though the lamp is off.
She likes the colour but dislikes the shape.
This road is shorter, although it is usually more crowded.
The cheaper shoes lasted longer than the expensive ones.
He speaks softly when he is angry and loudly when he is excited.
The book was difficult at first and easier towards the end.
I brought enough cups but forgot the plates.
They agreed on the problem but disagreed about the solution.
The door opens inwards, not outwards.
I thought the train was late, but I had arrived early.
She remembered his face without remembering his name.
The weather improved as we climbed higher.
It looks old from outside, but the rooms are newly painted.
We expected a small gathering and found a crowded hall.
The story sounds unlikely, yet several people witnessed it.''',
}
CASES = []
for group, block in GROUPS.items():
    texts = block.splitlines()
    assert len(texts) == 16, (group, len(texts))
    CASES.extend({'id': f'{group}-{i+1:02}', 'group': group, 'text': s} for i, s in enumerate(texts))

subjects = [
    'My older brother', 'My younger sister', 'Our neighbour', 'The school principal',
    'The bus driver', 'A visiting student', 'My grandmother', 'The shopkeeper',
    'The village carpenter', 'A friend from Srinagar',
]
actions = [
    'promised to return the borrowed books before the library closed',
    'forgot to bring the warm clothes that we had packed yesterday',
    'asked whether anyone had seen a small brown wallet near the gate',
    'refused to leave until the missing child had been found',
    'suggested that we take the longer road beside the river',
    'explained why the meeting could not begin before eleven',
    'remembered to buy bread but forgot the milk and eggs',
    'offered to help repair the broken chairs in the courtyard',
    'wondered if the people waiting outside had been invited',
    'decided to wait for the rain to stop before walking home',
]
for i, subject in enumerate(subjects):
    for j, action in enumerate(actions):
        for variant in range(2):
            text = f'{subject} {action}.' if variant == 0 else f'I was surprised when {subject[0].lower() + subject[1:]} {action}, because nobody had asked for help.'
            CASES.append({'id': f'composed-{i:02}-{j:02}-{variant}', 'group': 'composed', 'text': text})

openings = [
    'When we arrived at the village, the road was almost empty. A shopkeeper told us that most people had gone to a wedding.',
    'My grandmother asked me to write down an old story she remembered from her childhood. She paused often to explain the names of places.',
    'The children wanted to play outside, but dark clouds were gathering above the mountains. Their teacher suggested waiting a little longer.',
    'I found a notebook on the bus and looked for a name on its first page. There was only a drawing of a house beside a river.',
    'We had planned to finish the repairs before winter. Unfortunately, the timber arrived late and several tools were missing.',
    'The letter arrived three weeks after it was sent. By then, the person it was addressed to had moved to another town.',
    'At the market, two sellers were arguing about the price of apples. A customer quietly put down her basket and walked away.',
    'My sister wanted to surprise our parents with a visit. She asked me not to mention that she had already bought her ticket.',
]
endings = [
    'I was not sure what to do next, so I called my uncle and explained the situation. He listened carefully before suggesting that we speak to the neighbours.',
    'Nobody wanted to make a decision without hearing the whole story. We sat down together, shared some tea, and tried to understand what had happened.',
    'It was getting late, and we still had a long journey ahead of us. We agreed to return the following morning with enough time to finish everything properly.',
    'What seemed like a simple matter became more complicated as we learned the details. I wrote them down so that I would not confuse the dates or the names.',
    'I remember feeling both worried and hopeful. Even though we did not have an immediate answer, everyone was willing to help and nobody was left alone.',
]
for i, start in enumerate(openings):
    for j, end in enumerate(endings):
        CASES.append({'id': f'new_paragraph-{i:02}-{j:02}', 'group': 'new_paragraph', 'text': f'{start} {end}'})

assert len(CASES) == 400
assert len({c['text'] for c in CASES}) == 400
assert not ({c['text'] for c in CASES} & {c['text'] for c in OLD_CASES})
assert all(0 < len(c['text']) <= 500 for c in CASES)
path = ROOT / 'artifacts/expanded-corpus.json'
path.write_text(json.dumps(CASES, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'{len(CASES)} new inputs; {sum(len(c["text"]) for c in CASES)} characters; max {max(len(c["text"]) for c in CASES)} characters')
