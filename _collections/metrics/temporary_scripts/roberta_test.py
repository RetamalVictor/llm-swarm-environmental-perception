"""
NLI debug (DeBERTa cross-encoder). Edit the three blocks below, then run:

  python metrics/roberta_test.py

Label order: 0=contradiction, 1=entailment, 2=neutral (same as roberta_two).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

LABELS = ("contradiction", "entailment", "neutral")
_ENTAILMENT_IDX = 1

_model: CrossEncoder | None = None

# ---------------------------------------------------------------------------
# 1) Ground truth claims   2) Comm observation   3) Non-comm observation
#    (paste / edit as needed)
# ---------------------------------------------------------------------------

GROUND_TRUTH_FACTS = [
    "A mountain with snowy peaks is present.",
    "A stone bridge is present.",
    "A blue river is present.",
    "A large green tree is present.",
    "A dark volcano with orange lava is present.",
    "A red castle is present.",
    "A blue moat surrounds the castle.",
    "A green hedge maze is present.",
    "A group of four people is present.",
    "A gray parking lot is present.",
    "Six cars are present in the parking lot.",
    "A football stadium is present.",
    "A windmill is present.",
    "A fenced graveyard with tombstones is present.",
    "A colorful hot air balloon is present.",
    "A dense patch of trees is present.",
    "A blue stone fountain is present.",
    "A fenced zoo enclosure is present.",
    "A giraffe is present in the enclosure.",
    "An elephant is present in the enclosure.",
    "A lion is present in the enclosure.",
]

# One paragraph to score (e.g. last snapshot from one robot, or merged sample)
COMMUNICATION_OBSERVATION = [
        "A lava flow is present. The lava flow is orange. A stadium is present. The stadium has blue seats. A red car is present. first run. no observations so far.",
        "A castle is present. The castle is red. A river is present. The river is blue. A flag is present. The flag is red. A lava flow is present. The lava flow is orange. A stadium is present. The stadium has blue seats. A red car is present. first run. no observations so far.",
        "A green gradient is present. The green gradient has a lighter shade at the top. The green gradient has a darker shade at the bottom. Small white shapes are present. The small white shapes are scattered. A lava flow is present. The lava flow is orange. A stadium is present. The stadium has blue seats. A red car is present. Trees are present. The trees have green foliage. Water is present. The water has a blue color. A path is present. The path has a tan color. A castle is present. The castle is red. A river is present. The river is blue. A flag is present. The flag is red. A mountain range is present. The mountain range has snow. The mountain range has rocky slopes. first run. no observations so far.",
        "A castle is present. The castle is red. A flag is present. The flag is red. Water is present. The water has a blue color. A path is present. The path has a tan color. Trees are present. The trees have green foliage. A lava flow is present. The lava flow is orange. A stadium is present. The stadium has blue seats. A red car is present. A yellow car is present. A blue car is present. A green car is present. A river is present. The river is blue. A mountain range is present. The mountain range has snow. The mountain range has rocky slopes. A person is present. The person has a pink shirt. The person has a blue shirt. The person has a green vest. The person has a red shirt. A green gradient is present. The green gradient has a lighter shade at the top. The green gradient has a darker shade at the bottom. Small white shapes are present. The small white shapes are scattered. A parking lot is present. A beach ball is present. The beach ball has multiple. first run. no observations so far.",
        "A volcano is present. The volcano has a dark grey exterior. Lava is present. The lava is orange. A castle is present. The castle is red. A flag is present. The flag is red. Water is present. The water has a blue color. A path is present. The path has a tan color. Trees are present. The trees have green foliage. A lava flow is present. The lava flow is orange. A stadium is present. The stadium has blue seats. A red car is present. A yellow car is present. A blue car is present. A green car is present. A river is present. The river is blue. A mountain range is present. The mountain range has snow. The mountain range has rocky slopes. A person is present. The person has a pink shirt. The person has a blue shirt. The person has a green vest. The person has a red shirt. A green gradient is present. The green gradient has a lighter shade at the top. The green gradient has a darker shade at the bottom. Small white shapes are present. The small white shapes are scattered. A parking lot is present. A beach ball is present. The beach ball has multiple.",
        "A tombstone is present. The tombstone has a grey color. A grave is present. A cemetery is present. The cemetery has a fence. The fence is black. A castle is present. A flag is present. Water is present. A path is present. Trees are present. A lava flow is orange. A stadium is present. A red car is present. A yellow car is present. A blue car is present. A green car is present. A river is present. A mountain range is present. A person is present. A parking lot is present. A beach ball is present. A hedge maze is present. A rock is present. A volcano is present. A scoreboard is present. A giraffe is present. A fence is present. A building is present. The volcano has a dark grey exterior. Lava is present. The lava is orange. The castle is red. The flag is red. The water has a blue color. The path has a tan color. The trees have green foliage. The stadium has blue seats. The river is blue. The mountain range has snow.",
        "The fountain has water. The fountain has stone. A castle is present. A flag is present. Water is present. A path is present. Trees are present. A lava flow is orange. A stadium is present. A red car is present. A yellow car is present. A blue car is present. A river is present. A mountain range is present. A person is present. A parking lot is present. A beach ball is present. A hedge maze is present. A rock is present. A volcano is present. A scoreboard is present. A giraffe is present. A fence is present. A building is present. The volcano has a dark grey exterior. Lava is present. The lava is orange. The castle is red. The flag is red. The water has a blue color. The path has a tan color. The trees have green foliage. The stadium has blue seats. The river is blue. The mountain range has snow. The mountain range has rocky slopes. A bridge is present. A fountain is present. A tunnel is present. A green car is present.",
        "A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. The fountain has water. The fountain has stone. A castle is present. A flag is present. Water is present. A path is present. Trees are present. A lava flow is orange. A stadium is present. A river is present. A mountain range is present. A person is present. A beach ball is present. A hedge maze is present. A rock is present. A volcano is present. A scoreboard is present. A giraffe is present. A fence is present. A building is present. The volcano has a dark grey exterior. Lava is present. The lava is orange. The castle is red. The flag is red. The water has a blue color. The path has a tan color. The trees have green foliage. The stadium has blue seats. The river is blue. The mountain range has snow. The mountain range has rocky slopes. A bridge is present. A fountain is present. A tunnel is present.",
        "A tower is present. The tower has red bricks. A moat is present. The moat has blue water. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. The fountain has water. The fountain has stone. A castle is present. A flag is present. Water is present. A path is present. Trees are present. A lava flow is orange. A stadium is present. A river is present. A mountain range is present. A person is present. A beach ball is present. A hedge maze is present. A rock is present. A volcano is present. A scoreboard is present. A giraffe is present. A fence is present. A building is present. The volcano has a dark grey exterior. Lava is present. The lava is orange. The castle is red. The flag is red. The water has a blue color. The path has a tan color. The trees have green foliage. The stadium has blue seats. The river is blue. The mountain range has snow.",
        "The tree has a brown trunk. The tree has green leaves. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. The fountain has water. A castle is present. A flag is present. Water is present. A path is present. Trees are present. A lava flow is orange. A stadium is present. A river is present. A mountain range is present. A person is present. A beach ball is present. A hedge maze is present. A rock is present. A volcano is present. A scoreboard is present. A giraffe is present. A fence is present. A building is present. The volcano has a dark grey exterior. Lava is present. The lava is orange. The castle is red. The flag is red. The water has a blue color. The path has a tan color. The trees have green foliage. The stadium has blue seats. The river is blue. The mountain range has snow. The mountain range has rocky slopes. A bridge is present. A tunnel is present.",
        "A bridge is present. The bridge has stone. A river is present. The river is blue. Grass is present. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A fountain has blue water. A stadium has blue seats. A river is blue. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano has a dark rocky exterior. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge has stone. A soccer field has white lines. Lights are present. An elephant has grey skin. A lion has a mane. A windmill has. Trees have green foliage. A bush is present. The tree has a brown trunk. The tree has green leaves. Water is present. A path is present. Trees are present. A stadium is present. A mountain range is present.",
        "A cave entrance is present. The cave entrance has an archway. Trees are present. The trees have green foliage. Rocks are present. The mountain range has snow. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A fountain has blue water. A stadium has blue seats. A river is blue. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano has a dark rocky exterior. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge has stone. A soccer field has white lines. Lights are present. An elephant has grey skin. A lion has a mane. Grass is present. A windmill has four blades. Trees have green foliage. A bush is present. The tree has a brown trunk. Water is present. A path is present. A stadium is present. A river is present.",
        "A person is present. The person has a blue shirt. The person has blue pants. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is present. A fountain has blue water. A stadium has blue seats. A river is present. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge is present. A soccer field has white lines. Lights are present. An elephant has grey skin. A lion has a mane. Grass is present. A windmill has four blades. Trees have green foliage. A bush is present. The tree has a brown trunk. Water is present. A path is present. Trees are present. A scoreboard is present. A moat is present. A tunnel is present. A hot air balloon has a basket.",
        "A windmill is present. The windmill has four blades. A building is present. The building has windows. Grass is present. A path is present. Trees are present. A bush is present. The tree has a brown trunk. Water is present. A moat is present. A tunnel is present. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is present. A fountain has blue water. A stadium has blue seats. A river is present. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge is present. A soccer field has white lines. Lights are present. An elephant has grey skin. A lion has a mane. A windmill has four blades. Trees have green foliage. A scoreboard is present.",
        "A hot air balloon is present. The hot air balloon has a basket. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A fountain has blue water. A stadium has blue seats. A river is present. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge is present. A soccer field has white lines. Lights are present. An elephant is present. A lion has a mane. Grass is present. A windmill has four blades. Trees have green foliage. A bush is present. The tree has a brown trunk. Water is present. A path is present. Trees are present. A scoreboard is present. A moat is present. A tunnel is present. A hot air balloon has a basket. A parrot is present.",
        "A fountain has blue water. Trees are present. A hot air balloon is present. The hot air balloon has a basket. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river is present. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge is present. A soccer field has white lines. Lights are present. An elephant is present. A lion has a mane. Grass is present. A windmill has four blades. Trees have green foliage. A bush is present. The tree has a brown trunk. Water is present. A path is present. A scoreboard is present. A moat is present. A tunnel is present. A hot air balloon has a basket. A parrot is present.",
        "An elephant is present. A giraffe is present. The giraffe has spots. A lion is present. The lion has a mane. A fence is present. The fence has wooden posts. Grass is present. A path is present. A fountain has blue water. Trees are present. A hot air balloon is present. The hot air balloon has a basket. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river is present. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A giraffe has spots. A fence has wooden posts. A building has windows. A bridge is present. A soccer field has white lines. Lights are present. A lion has a mane. A windmill has four blades. Trees have green foliage. A bush is present. The tree has a brown trunk.",
        "A stadium is present. The stadium has blue seats. The stadium has red seats. A soccer field is present. The soccer field has white lines. Lights are present. A fence is present. The fence has wooden posts. Trees are present. Trees have green foliage. A bush is present. An elephant is present. A giraffe is present. The giraffe has spots. A lion is present. The lion has a mane. Grass is present. A path is present. A fountain has blue water. A hot air balloon is present. The hot air balloon has a basket. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river is present. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A giraffe has spots. A fence has wooden posts. A building has windows.",
        "A dragon is present. A volcano has lava. A river is present. A tower is present. A tower has red accents. A dragon has black scales. A giraffe has spots. A lion has a mane. A fence has wooden posts. Grass is present. A path has a tan color. A fountain has blue water. Trees have green foliage. A hot air balloon has a basket. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle is red. A flag is red. A stadium has blue seats. A river has blue water. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A building has windows. A bridge is present. A soccer field has white lines. Lights are present. A windmill has four blades. A bush is present. A golf ball is present. A golf bag has clubs. A scoreboard is. A stadium is present.",
        "A tree is present. The tree has green foliage. A giraffe has spots. A lion has a mane. A fence has wooden posts. Grass is present. A path has a tan color. A fountain has blue water. Trees have green foliage. A hot air balloon has a basket. A yellow car is present. A blue car is present. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river has blue water. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A building has windows. A bridge has stone. Lights are on poles. A windmill has four blades. A golf ball is present. A scoreboard is. A moat has blue water. A dragon is present. People are present. A volcano has lava. A river is present. A tower is present. A tower has red accents. A dragon has black scales.",
        "A bridge is present. A river is present. The bridge has stone. The river has blue water. A giraffe has spots. A lion has a mane. A fence has wooden posts. Grass is present. A path has a tan color. A fountain has blue water. Trees have green foliage. A hot air balloon has a basket. A yellow car is present. A blue car has wheels. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river has blue water. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A building has windows. A bridge has stone. Lights are on poles. A windmill has four blades. A golf ball is present. A scoreboard is. A moat has blue water. A dragon is present. People are present. A tower is present. A tower has red accents. A dragon has black scales.",
        "A mountain range is present. The mountain range has snow. Trees are present. A tunnel is present. A giraffe has spots. A lion has a mane. A fence has wooden posts. Grass is present. A path has a tan color. A fountain has blue water. Trees have green foliage. A hot air balloon has a basket. A yellow car is present. A blue car has wheels. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river has blue water. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A building has windows. A bridge has stone. Lights are on poles. A windmill has four blades. A golf ball is present. A scoreboard is. A moat has blue water. A dragon is present. People are present. A tower is present. A tower has red accents. A dragon has black scales.",
        "A castle is red. A flag is red. A moat has blue water. A hedge maze is present. A path has a tan color. A tree is present. A giraffe has spots. A lion has a mane. A fence has wooden posts. Grass is present. A fountain has blue water. Trees have green foliage. A hot air balloon has a basket. A yellow car is present. A blue car has wheels. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A stadium has blue seats. A river has blue water. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A rock is present. A volcano is present. A building has windows. A bridge has stone. Lights are on poles. A windmill has four blades. A golf ball is present. A scoreboard is. A dragon is present. People are present. A tower is present. A tower has red accents. A dragon has black scales. An elephant has grey skin. A mountain range is present.",
        "A fence is present. A tombstone is present. A tree is present. A cross is present. A graveyard is present. The fence has metal bars. The tombstone has engravings. The tree has bare branches. A giraffe has spots. A lion has a mane. A fence has wooden posts. Grass is present. A path has a tan color. A fountain has blue water. Trees have green foliage. A hot air balloon has a basket. A yellow car is present. A blue car has wheels. A red car is present. A green car is present. A parking lot is present. A castle has red roofs. A flag is red. A stadium has blue seats. A river has blue water. A mountain range has snow. A person has a pink shirt. A beach ball has multiple colors. A lava flow is orange. A hedge maze is present. A rock is present. A volcano is present. A building has windows. A bridge has stone. Lights are on poles. A windmill has four blades. A golf ball is present. A scoreboard is. A moat has blue water. A dragon is present."
    ]

NON_COMMUNICATION_OBSERVATION = [
        "A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A bridge is present. The bridge has stone. Water is present. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A person is present. The person has brown hair. A bridge is present. The bridge has stone. Water is present. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A castle is present. The castle has red brick. A moat is present. The moat has water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A person is present. The person has brown hair. A bridge is present. The bridge has stone. Water is present. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A car is present. The car has red paint. The car has blue paint. The car has green paint. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. Water is present. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A car is present. The car has red paint. The car has blue paint. The car has green paint. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. Water is present. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A car is present. The car has red paint. The car has blue paint. The car has green paint. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run. no observations so far.",
        "A mountain is present. The mountain has snow. A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. A volcano is present. The volcano has lava. Rocks are present. The rocks are gray. first run.",
        "A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. The person has brown hair. A bridge is present. The bridge has stone. A volcano is present. The volcano has lava. Rocks are present.",
        "A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present.",
        "A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots. An elephant is present. The elephant is gray. A fence is present. The fence has wood.",
        "A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint. The car has green paint. The car has red paint. A fountain is present. The fountain has stone. Water is present. The water is blue. A giraffe is present. The giraffe has yellow spots.",
        "A fence is present. The fence has posts. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A tree is present. The tree has brown bark. A maze is present. The maze has hedges. The hedges are green. A car is present. The car has yellow paint. The car has blue paint.",
        "A graveyard is present. The graveyard has tombstones. A fence is present. The fence has posts. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A maze is present. The maze has hedges. The hedges are green. A car is present.",
        "A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A fence is present. The fence has posts. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow. A maze is present. The maze has hedges.",
        "A maze is present. The maze has hedges. A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A fence is present. The fence has posts. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present. The hot air balloon has yellow. The hot air balloon has blue. The hot air balloon has red. A forest is present. The forest has green trees. A cave is present. The cave has an entrance. A mountain is present. The mountain has snow.",
        "A car is present. The car is red. The car is green. The car is blue. The car is light blue. A parking lot is present. The parking lot has asphalt. A beach ball is present. The beach ball has red. A maze is present. The maze has hedges. A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A fence is present. The fence has posts. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present. The castle has red brick. A moat is present. The moat has water. A person is present. A hot air balloon is present.",
        "A hot air balloon is present. The hot air balloon has a basket. A person is present. The person has a blue helmet. The person has a pink shirt. A car is present. The car is red. The car is green. The car is blue. The car is light blue. A parking lot is present. The parking lot has asphalt. A beach ball is present. The beach ball has red. A maze is present. The maze has hedges. A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A fence is present. The fence has posts. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present. The stadium has blue seats. The stadium has red seats. Lights are present. The lights are on poles. A castle is present.",
        "A windmill is present. The windmill has a blue roof. The windmill has white sails. A tombstone is present. The tombstone has a grey color. A fence is present. The fence has posts. A hot air balloon is present. The hot air balloon has a basket. A person is present. The person has a blue helmet. The person has a pink shirt. A car is present. The car is red. The car is green. The car is blue. The car is light blue. A parking lot is present. The parking lot has asphalt. A beach ball is present. The beach ball has red. A maze is present. The maze has hedges. A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water. A stadium is present.",
        "A person is present. The person has brown hair. The person has a pink shirt. A windmill is present. The windmill has a blue roof. The windmill has white sails. A tombstone is present. The tombstone has a grey color. A fence is present. The fence has posts. A hot air balloon is present. The hot air balloon has a basket. The person has a blue helmet. A car is present. The car is red. The car is green. The car is blue. The car is light blue. A parking lot is present. The parking lot has asphalt. A beach ball is present. The beach ball has red. A maze is present. The maze has hedges. A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone. A bridge is present. The bridge has stone. A river is present. The river has blue water.",
        "A bridge is present. The bridge has stone. A river is present. The river has blue water. A person is present. The person has brown hair. The person has a pink shirt. A windmill is present. The windmill has a blue roof. The windmill has white sails. A tombstone is present. The tombstone has a grey color. A fence is present. The fence has posts. A hot air balloon is present. The hot air balloon has a basket. The person has a blue helmet. A car is present. The car is red. The car is green. The car is blue. The car is light blue. A parking lot is present. The parking lot has asphalt. A beach ball is present. The beach ball has red. A maze is present. The maze has hedges. A flower is present. The flower is white. A graveyard is present. The graveyard has tombstones. A tree is present. The tree has brown bark. A gate is present. A statue is present. A building is present. The building has a roof. A path is present. The path has stone."
    ]


def get_model(model_name: str | None = None) -> CrossEncoder:
    global _model
    if _model is None:
        name = model_name or MODEL_NAME
        print(f"Loading {name}...")
        _model = CrossEncoder(name)
    return _model


def nli_two_sentences(
    premise: str,
    hypothesis: str,
    model: CrossEncoder | None = None,
    apply_softmax: bool = True,
) -> dict[str, Any]:
    m = model or get_model()
    logits = m.predict([(premise, hypothesis)])[0]
    logits = np.asarray(logits, dtype=np.float64)
    if apply_softmax:
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        idx = int(np.argmax(probs))
    else:
        probs = None
        idx = int(np.argmax(logits))

    return {
        "label": LABELS[idx],
        "label_index": idx,
        "logits": logits,
        "probs": probs,
    }


def nli_claim_vs_paragraph(
    claim: str,
    observation: str,
    model: CrossEncoder | None = None,
    apply_softmax: bool = True,
) -> dict[str, Any]:
    m = model or get_model()
    claim = claim.strip()
    observation = observation.strip()
    if not claim or not observation:
        return {
            "entailed": False,
            "per_sentence": [],
            "best_entailment_sentence": None,
        }

    sentences = [s.strip() for s in sent_tokenize(observation) if s.strip()]
    if not sentences:
        return {
            "entailed": False,
            "per_sentence": [],
            "best_entailment_sentence": None,
        }

    pairs = [(sent, claim) for sent in sentences]
    logits_batch = m.predict(pairs)
    per_sentence: list[dict[str, Any]] = []
    entailed = False
    best_sent: str | None = None
    best_entail_p = -1.0

    for sent, row in zip(sentences, logits_batch):
        logits = np.asarray(row, dtype=np.float64)
        if apply_softmax:
            e = np.exp(logits - np.max(logits))
            probs = e / e.sum()
            idx = int(np.argmax(probs))
            ent_p = float(probs[_ENTAILMENT_IDX])
        else:
            probs = None
            idx = int(np.argmax(logits))
            ent_p = float("nan")

        label = LABELS[idx]
        if idx == _ENTAILMENT_IDX:
            entailed = True
            if apply_softmax and ent_p > best_entail_p:
                best_entail_p = ent_p
                best_sent = sent

        per_sentence.append(
            {
                "sentence": sent,
                "label": label,
                "label_index": idx,
                "logits": logits,
                "probs": probs,
            }
        )

    if entailed and best_sent is None:
        for row in per_sentence:
            if row["label_index"] == _ENTAILMENT_IDX:
                best_sent = row["sentence"]
                break

    return {
        "entailed": entailed,
        "per_sentence": per_sentence,
        "best_entailment_sentence": best_sent,
    }


def entailed_per_claim_mask(
    claims: list[str],
    observations: list[str],
    model: CrossEncoder | None = None,
) -> list[bool]:
    m = model or get_model()
    if not claims:
        return []
    all_text = " ".join((obs or "").strip() for obs in observations if obs and obs.strip())
    sentences = [s.strip() for s in sent_tokenize(all_text) if s.strip()]
    if not sentences:
        return [False] * len(claims)

    pairs = [(sent, c) for c in claims for sent in sentences]
    logits = m.predict(pairs)
    preds = np.argmax(np.asarray(logits), axis=1)
    n_claims = len(claims)
    n_sent = len(sentences)
    mat = preds.reshape(n_claims, n_sent)
    return (mat == _ENTAILMENT_IDX).any(axis=1).tolist()


def coverage_and_missing(
    claims: list[str],
    observations: list[str],
    label: str,
    model: CrossEncoder | None = None,
) -> tuple[float, list[str]]:
    mask = entailed_per_claim_mask(claims, observations, model)
    n = len(claims)
    cov = float(sum(mask)) / n if n else 0.0
    missing = [claims[i] for i in range(n) if not mask[i]]
    print(f"\n--- {label} ---")
    print(f"Observations used: {len(observations)}")
    print(f"Coverage: {cov:.4f}  ({int(sum(mask))}/{n} claims entailed)")
    print("Missing (no entailment from any sentence):")
    if not missing:
        print("  (none)")
    else:
        for c in missing:
            print(f"  - {c}")
    return cov, missing


if __name__ == "__main__":
    m = get_model()
    coverage_and_missing(GROUND_TRUTH_FACTS, COMMUNICATION_OBSERVATION, "Communication", m)
    coverage_and_missing(GROUND_TRUTH_FACTS, NON_COMMUNICATION_OBSERVATION, "Non-communication", m)
