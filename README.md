<h1>EMADropFormer: Speech Emotion Recognition</h1>

<h2>Project Overview</h2>
<p>
This project implements three Speech Emotion Recognition systems on the RAVDESS dataset:
</p>

<ul>
<li>MFCC + MLP Baseline</li>
<li>Wav2Vec2 Frozen Baseline</li>
<li>EMADropFormer (Proposed)</li>
</ul>

<h2>Dataset</h2>
<p>RAVDESS dataset with 24 actors and 1440 wav files.</p>

<h2>Run Commands</h2>

<pre>
python baseline_mfcc.py
python baseline_wav2vec2.py
python emadropformer_final.py
python compare_results.py
</pre>

<h2>Results Comparison</h2>

<h3>A. Our Experimental Results on RAVDESS</h3>

<table border="1">
<tr>
<th>Model</th>
<th>Dataset</th>
<th>Accuracy</th>
<th>Weighted F1</th>
</tr>

<tr>
<td>MFCC + MLP</td>
<td>RAVDESS</td>
<td>40.97%</td>
<td>39.94%</td>
</tr>

<tr>
<td>Wav2Vec2 Frozen</td>
<td>RAVDESS</td>
<td>62.50%</td>
<td>62.51%</td>
</tr>

<tr>
<td><b>EMADropFormer (Ours)</b></td>
<td>RAVDESS</td>
<td><b>82.00%</b></td>
<td><b>82.00%</b></td>
</tr>
</table>

<h3>B. Published Reference Results (Reported by Original Papers)</h3>

<table border="1">
<tr>
<th>Model</th>
<th>Dataset</th>
<th>Metric</th>
<th>Reported Score</th>
</tr>

<tr>
<td>DropFormer</td>
<td>IEMOCAP</td>
<td>WA</td>
<td>75.29</td>
</tr>

<tr>
<td>DropFormer</td>
<td>IEMOCAP</td>
<td>UA</td>
<td>76.60</td>
</tr>

<tr>
<td>DropFormer</td>
<td>MELD</td>
<td>WF1</td>
<td>49.25</td>
</tr>
</table>

<p>
Published scores are taken from the original DropFormer paper and evaluated on different datasets/protocols, so they are not directly comparable to RAVDESS.
</p>

<h3>C. Observation</h3>

<ul>
<li>EMADropFormer achieved strong performance on RAVDESS.</li>
<li>Pretrained speech representations significantly improved accuracy.</li>
<li>Emotion-guided attention improved discriminative learning.</li>
</ul>

<h2>Dataset Contextual Comparison</h2>

<table border="1">
<tr>
<th>Dataset</th>
<th>Type</th>
<th>Classes</th>
<th>Difficulty</th>
<th>Notes</th>
</tr>

<tr>
<td>RAVDESS</td>
<td>Acted studio speech</td>
<td>8</td>
<td>Moderate</td>
<td>Clean recordings, clear emotions</td>
</tr>

<tr>
<td>IEMOCAP</td>
<td>Scripted + improvised dialogue</td>
<td>4 commonly used</td>
<td>High</td>
<td>Natural variability, speaker interaction</td>
</tr>

<tr>
<td>MELD</td>
<td>Conversation from TV dialogues</td>
<td>7</td>
<td>Very High</td>
<td>Multi-speaker context, noisy emotion cues</td>
</tr>
</table>

<h2>Expected Generalization</h2>

<p>
RAVDESS is cleaner and more controlled than IEMOCAP and MELD. Therefore, higher scores are generally expected on RAVDESS. 
More challenging conversational datasets such as IEMOCAP and MELD usually produce lower metrics due to spontaneous speech, contextual ambiguity, and speaker variation.
</p>

<p>
Given the strong 82% result on RAVDESS, the proposed EMADropFormer is expected to remain competitive on more difficult datasets after task-specific fine-tuning.
</p>

<h2>Key Contributions</h2>

<ul>
<li>Emotion-guided token weighting</li>
<li>Partial fine-tuning of pretrained speech model</li>
<li>Mean + Max temporal pooling</li>
<li>Competitive SER performance</li>
</ul>

<h2>Author</h2>
<p>Kapil, Bhagwan, Suhani, Kowshika</p>