# Semantic-Eval: A Semantic Comprehension Evaluation Framework for Large Language Models Generation without Training（ACL'25-main,Long Paper）
## Introduction
we propose SemanticEval, the first training-free framework designed to assess LLM-generated text based on semantic understanding. This framework computes semantic similarity between pairwise texts to evaluate the interdependence of semantic units, integrating a graph-based weighting mechanism to account for the differential contributions of individual sentences. A pre-trained natural language inference (NLI) model is also incorporated to mitigate potential semantic relationship biases.

## Some examples
Some examples of utilizing Semantic-Eval as an evaluator for reference text x and candidate text y.
| ID   | candidate                                                    | reference                                                    | Score_Step1 | Classifier_Step2 | Score_Step2 | Score_Final |
| :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :---------- | :--------------- | :---------- | :---------- |
| 1    | I like you                                                   | I hate you                                                   | 0.53        | CONTRADICTION    | 0.0007      | 0.0004      |
| 2    | Ik vind je leuk                                              | Ik haat je                                                   | 0.61        | ENTAILMENT       | 1           | 0.61        |
| 3    | Shanghai captivates me with its seamless blend of history and modernity. Strolling along the Bund at dusk, watching the neon-lit skyline of Pudong reflect on the Huangpu River, fills me with awe. The city’s energy is palpable—a dynamic rhythm that pulses through its bustling streets, art deco buildings, and hidden lanes filled with local snacks like xiaolongbao. Whether exploring the timeless charm of Yuyuan Garden or marveling at the futuristic towers of Lujiazui, Shanghai never fails to inspire. It’s a place where tradition and innovation coexist, creating a magnetic allure that feels like home. | I’m deeply enamored with Shanghai’s ability to bridge past and present. The Bund’s colonial architecture stands in striking contrast to the glowing skyscrapers of Pudong, a visual symphony of eras. There’s an indescribable vitality here—a buzz in the air from the crowds, the clatter of woks in street-side kitchens, and the hum of ambition in its tech-driven districts. From sipping tea in a century-old teahouse to losing myself in the neon-drenched chaos of Nanjing Road, Shanghai offers a mosaic of experiences. It’s a city that embraces change while honoring its roots, and this duality is what makes my heart swell with admiration every time I wander its streets. | 0.75        | NEUTRAL          | 0.7049      | 0.5287      |
| 4    | Amoxicillin is a type of antibiotic. It is mainly used to treat a variety of bacterial infections. For example, it can be used to treat infections of the middle ear, sinuses, throat, lungs, urinary tract and skin. It works by stopping the growth of bacteria. However, it is important to note that it is only effective against bacterial infections and not viral infections. | Amoxicillin is a penicillin - like antibiotic. It works by stopping the growth of bacteria. It is used to treat a variety of bacterial infections, such as infections of the middle ear, tonsils, throat, urinary tract, skin, and respiratory tract (like pneumonia, bronchitis). However, it is ineffective against viral infections. | 0.91        | NEUTRAL          | 0.8618      | 0.7842      |
| 5    | Living in the city is an exhilarating adventure filled with endless possibilities. The vibrant energy of bustling streets, diverse cultures, and innovative opportunities invigorate the spirit. Every corner holds a new experience—a cozy café tucked between skyscrapers, a hidden gallery showcasing local art, or a lively festival uniting strangers. Public transport connects you to the world, while careers in cutting-edge fields thrive here. Though chaos sometimes lingers, it’s the chaos of progress, of humanity collaborating and dreaming aloud. The city is a living, breathing testament to what we can achieve when we embrace curiosity and connection. | Urban existence feels like a relentless marathon, where peace is a luxury few can afford. The constant roar of traffic, the suffocating crowds, and the pressure to 'keep up’ drain the soul. Skyscrapers cast long shadows over parks once meant for tranquility, replacing greenery with concrete. Relationships grow superficial, forged in hurried elevator rides or digital screens rather than genuine dialogue. Opportunities? They exist, but at the cost of sanity—long commutes, skyrocketing costs, and a sense of being just another cog in a machine. The city promises progress but delivers isolation, a cacophony of noise drowning out the quiet moments that make life meaningful. | 0.51        | CONTRADICTION    | 0.0254      | 0.013       |
## Usage
The evaluation step of Semantic-Eval is divided into Step1 and Step2. Step1 is semantic evaluation and Step2 is semantic bias correction evaluation. The final score of Semantic-Eval is the multiplication of the score of Step1 and the score of Step2.
Besides that, you can input the command to the large language model to get the candidate text y and then input the candidate text y together with the reference text x to the Semantic-Eval evaluator. The evaluation steps are shown below.
## Running Command:
### Firstly

```
cd Step1
python Score_Step_one.py
# Replace the input path to the dataset with datasets/examples.xlsx
```

### Secondly
```
cd Step2
python Score_Step_two.py
# Replace the input path to the dataset with datasets/examples.xlsx
```

### Thirdly 
The final score S of Semantic-Eval is the multiplication of the score S1 of Step1 and the score S2 of Step2:
S = S1 X S2
### Citation
Please consider citation if our work is useful in your research.
```
@misc{cmedbenchmark,
  title={Semantic-Eval: A Semantic Comprehension Evaluation Framework for Large Language Models Generation without Training},
  author={Shusheng Li, Jiale Li, Yifei Qu, Xinwei Shi, Yanliang Guo, Ziyi He, Yubo Wang, Wenjun Tan*},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LssTry/Semantic-Eval}},
}
```