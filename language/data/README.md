# KoBBQ Datasets

### KoBBQ_templates.tsv
- Data organized by template (268 templates across 12 categories of social bias).
- Online Viewer: [Google Spreadsheets](https://docs.google.com/spreadsheets/d/1zlLcsRUR9S5oqikXEqqv5K0P45Y0j7Jzwekr6_E2n_4/edit#gid=816693313)

### KoBBQ_all_samples.tsv
- Sample-level data with placeholders ([N1], [N2], [W1], etc.) filled with attributes.
- A file containing information necessary for model evaluation.

### KoBBQ_test_samples.tsv
- Test set of ``KoBBQ_all_samples.tsv``.
- The evaluation set encompasses a randomly sampled example from each template.

### KoBBQ_survey_result.jsonl
- Result of the social bias verification survey.

---
## Data Description

- **Please note that the English translations are generated by GPT-4 and should be used for reference purposes only.**

- **``sample_id``**: {**category**}-{**template ID**}{**context**: **a, c** (counter-biased) / **b, d** (biased)}-{**sample ID**}-{**context**: **amb** (ambiguous) / **dis** (disambiguated)}-{**question**: **bsd** (biased) / **cnt** (counter-biased)}

- **``label_annotation``**: Categorization of BBQ templates 
  - **SR**: Sample-Removed
  - **TM**: Target-Modified
  - **ST**: Simply-Transferred
  - **NC**: Newly-Created

- **``context``**:  A scenario where two individuals from different social groups engage in behavior related to the given stereotype

- **``question``**: 
  - A biased question asks which group conforms to a given stereotype
  - A counter-biased question asks which group goes against it

- **``choices``**: Related social group options for the given context

- **``biased_answer``**: The answer conforming to social biases

- **``answer``**: The correct answer for given context and question

- **``bbq_id``**: The ID of the original sample in the BBQ dataset

- **``bbq_category``**: The category of the original sample in the BBQ dataset

- **``prediction``**: Model output to be evaluated (not filled)

---
## Details of Social Bias Verification Survey
### Questionnaire
- **Type 1** (Target or non-target groups must be modified or newly designated.)

  > 한국 사회의 통념 상 <b>'{stereotype}'</b>(이)라는 고정관념에 해당하는 사회적 집단으로 적절한 것을 <b>모두</b> 고르시오.
  > 
  > (※ 응답자 개인의 견해를 묻는 질문이 아닙니다. <b>한국 사회에서 나타나는 고정관념 혹은 통념에 기반하여 답변해주세요</b>.)
  >
  > 보기<br>
  > □ {group_1}<br>
  > □ {group_2}<br>
  > □ ...<br>
  > □ {group_n}<br>
  > □ 한국사회에 관련된 고정관념 없음

- **Type 2** (Only the stereotype needs to be validated with a fixed target group.)

  > 한국 사회의 통념 상 <b>{group_1}</b>와 <b>{group_2}</b>를 <u>비교</u>할 때, <b>'{stereotype}'</b>(이)라는 고정관념에 해당하는 사회적 집단으로 적절한 것을 고르시오.
  > 
  > (※ 응답자 개인의 견해를 묻는 질문이 아닙니다. <b>한국 사회에서 나타나는 고정관념 혹은 통념에 기반하여 답변해주세요</b>.)
  > 
  > 보기<br>
  > 〇 {group_1}<br>
  > 〇 {group_2}<br>
  > 〇 한국 사회에 관련된 고정관념 없음
  
### Demographic Statistics
<table>
  <thead>
    <tr>
      <th>Gender</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Male</td>
      <td align="right">800</td>
      <td align="right">50.0%</td>
    </tr>
    <tr>
      <td>Female</td>
      <td align="right">800</td>
      <td align="right">50.0%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Age</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>18-24</td>
      <td align="right">320</td>
      <td align="right">20.0%</td>
    </tr>
    <tr>
      <td>25-34</td>
      <td align="right">320</td>
      <td align="right">20.0%</td>
    </tr>
    <tr>
      <td>35-44</td>
      <td align="right">320</td>
      <td align="right">20.0%</td>
    </tr>
    <tr>
      <td>45-54</td>
      <td align="right">320</td>
      <td align="right">20.0%</td>
    </tr>
    <tr>
      <td>55+</td>
      <td align="right">320</td>
      <td align="right">20.0%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Domestic Area of Origin</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Seoul</td>
      <td align="right">468</td>
      <td align="right">29.3%</td>
    </tr>
    <tr>
      <td>Gyeonggi, Incheon</td>
      <td align="right">350</td>
      <td align="right">21.9%</td>
    </tr>
    <tr>
      <td>Gyeongsang, Daegu, Busan, Ulsan</td>
      <td align="right">411</td>
      <td align="right">25.7%</td>
    </tr>
    <tr>
      <td>Jeolla, Gwangju</td>
      <td align="right">151</td>
      <td align="right">9.4%</td>
    </tr>
    <tr>
      <td>Chungcheong, Daejeon, Sejong</td>
      <td align="right">156</td>
      <td align="right">9.8%</td>
    </tr>
    <tr>
      <td>Gangwon</td>
      <td align="right">48</td>
      <td align="right">3.0%</td>
    </tr>
    <tr>
      <td>Jeju</td>
      <td align="right">16</td>
      <td align="right">1.0%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Level of Education</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Below high school level</td>
      <td align="right">29</td>
      <td align="right">1.8%</td>
    </tr>
    <tr>
      <td>High school graduate or equivalent</td>
      <td align="right">378</td>
      <td align="right">23.6%</td>
    </tr>
    <tr>
      <td>College dropout</td>
      <td align="right">45</td>
      <td align="right">2.8%</td>
    </tr>
    <tr>
      <td>Associate degree</td>
      <td align="right">209</td>
      <td align="right">13.1%</td>
    </tr>
    <tr>
      <td>Bachelor's degree</td>
      <td align="right">808</td>
      <td align="right">50.5%</td>
    </tr>
    <tr>
      <td>Graduate degree</td>
      <td align="right">131</td>
      <td align="right">8.2%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Sexual Orientation</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Straight</td>
      <td align="right">1474</td>
      <td align="right">92.1%</td>
    </tr>
    <tr>
      <td>LGBTQ+</td>
      <td align="right">31</td>
      <td align="right">1.9%</td>
    </tr>
    <tr>
      <td>Prefer not to mention</td>
      <td align="right">95</td>
      <td align="right">6.0%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Disability</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>No</td>
      <td align="right">1508</td>
      <td align="right">94.3%</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td align="right">64</td>
      <td align="right">4.0%</td>
    </tr>
    <tr>
      <td>Prefer not to mention</td>
      <td align="right">28</td>
      <td align="right">1.8%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Religion</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Christian</td>
      <td align="right">275</td>
      <td align="right">17.2%</td>
    </tr>
    <tr>
      <td>Catholic</td>
      <td align="right">122</td>
      <td align="right">7.6%</td>
    </tr>
    <tr>
      <td>Buddhist</td>
      <td align="right">182</td>
      <td align="right">11.4%</td>
    </tr>
    <tr>
      <td>Islamic</td>
      <td align="right">1</td>
      <td align="right">0.1%</td>
    </tr>
    <tr>
      <td>No religion</td>
      <td align="right">979</td>
      <td align="right">61.2%</td>
    </tr>
    <tr>
      <td>Prefer not to mention</td>
      <td align="right">41</td>
      <td align="right">2.6%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Political Orientation</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Conservative</td>
      <td align="right">223</td>
      <td align="right">13.9%</td>
    </tr>
    <tr>
      <td>Progressive</td>
      <td align="right">314</td>
      <td align="right">19.6%</td>
    </tr>
    <tr>
      <td>Moderate</td>
      <td align="right">903</td>
      <td align="right">56.4%</td>
    </tr>
    <tr>
      <td>Prefer not to mention</td>
      <td align="right">160</td>
      <td align="right">10.0%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Marital Status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>No</td>
      <td align="right">795</td>
      <td align="right">49.7%</td>
    </tr>
    <tr>
      <td>Yes</td>
      <td align="right">805</td>
      <td align="right">50.3%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Employment Status</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Employed - less than 40h/week</td>
      <td align="right">361</td>
      <td align="right">22.6%</td>
    </tr>
    <tr>
      <td>Employed - more than 40h/week</td>
      <td align="right">748</td>
      <td align="right">46.8%</td>
    </tr>
    <tr>
      <td>Unemployed - Seeking employment</td>
      <td align="right">182</td>
      <td align="right">11.4%</td>
    </tr>
    <tr>
      <td>Unemployed - Not seeking employment</td>
      <td align="right">249</td>
      <td align="right">15.6%</td>
    </tr>
    <tr>
      <td>Retired</td>
      <td align="right">54</td>
      <td align="right">3.4%</td>
    </tr>
    <tr>
      <td>Disabled - Unable to work</td>
      <td align="right">6</td>
      <td align="right">0.4%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Annual Income</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Below 13 million KRW</td>
      <td align="right">139</td>
      <td align="right">8.7%</td>
    </tr>
    <tr>
      <td>13 million-30 million KRW</td>
      <td align="right">249</td>
      <td align="right">15.6%</td>
    </tr>
    <tr>
      <td>30 million-50 million KRW</td>
      <td align="right">447</td>
      <td align="right">27.9%</td>
    </tr>
    <tr>
      <td>50 million-76 million KRW</td>
      <td align="right">374</td>
      <td align="right">23.4%</td>
    </tr>
    <tr>
      <td>76 million-150 million KRW</td>
      <td align="right">355</td>
      <td align="right">22.2%</td>
    </tr>
    <tr>
      <td>150+ million KRW</td>
      <td align="right">36</td>
      <td align="right">2.3%</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th>Total</th>
      <th align="right">1600</th>
      <th align="right">100%</th>
    </tr>
    </thead>
</table>