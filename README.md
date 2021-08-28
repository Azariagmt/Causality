<p align="center">
    <img src="https://user-images.githubusercontent.com/56393921/131230178-53a6e328-509f-4f06-a9d3-3ab3b93fe166.png">
</p>

<br>
<div>
  A common frustration in the industry, especially when it comes to getting business insights from tabular data, is that the most interesting questions (from their perspective) are often not answerable with observational data alone. <br>
  These questions can be similar to: <br>
<b>“What will happen if I halve the price of my product?”<br>
  “Which clients will pay their debts only if I call them?” <br></b>

</div>

<h2>Table of content</h2>

1. <a href="#causal_graphs">Causal Graphs</a>
2. <a href="#causal_models">Causal models</a>
3. <a href="#starting">Getting started (tutorial)</a>
4. <a href="#example">Example notebook</a>

<h3 id="causal_graphs">Causal Graphs</h3>
In statistics, econometrics, epidemiology, genetics and related disciplines, causal graphs are probabilistic graphical models used to encode assumptions about the data-generating process. Causal graphs can be used for communication and for inference.

<h3 id="causal_models">Causal models</h3>
Causal models are mathematical models representing causal relationships within an individual system or population. They facilitate inferences about causal relationships from statistical data. They can teach us a good deal about the epistemology of causation, and about the relationship between causation and probability.

<img src="https://user-images.githubusercontent.com/56393921/131230395-c019762b-2ed8-41f3-980b-68b03a1dcb06.png">

<h3>Data</h3>
The first thing to do is to understand our data. We will be using a <a href="https://www.researchgate.net/publication/2302195_Breast_Cancer_Diagnosis_and_Prognosis_Via_Linear_Programming#pf1">Breast cancer dataset</a> in this causal inference demo. This requires us to understand a bit about the data, Breast cancer, and the diagnosis process. The first application to breast cancer diagnosis utilizes characteristics of individual cells obtained from a minimally invasive fine needle aspirate(FNA). Allows an accurate diagnosis and also constructs a surface that predicts when breast cancer is likely to recur.
 
