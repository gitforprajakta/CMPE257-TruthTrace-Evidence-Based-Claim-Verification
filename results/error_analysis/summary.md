# Error analysis

Reasons are assigned automatically by the first matching rule. `manual_reason` is left empty for the team to fill in.

## Counts

| model | reason | count |
| --- | --- | --- |
| tfidf_lr | other | 8 |
| tfidf_lr | conflicting_confusion | 6 |
| tfidf_lr | thin_evidence | 1 |
| sbert_lr | other | 8 |
| sbert_lr | conflicting_confusion | 3 |
| sbert_lr | thin_evidence | 3 |
| sbert_lr | short_claim | 1 |

## Examples

### tfidf_lr

- claim 34: true Supported; TF-IDF Refuted; SBERT Supported; reason other. The United States, New Zealand, U.K., Australia, and the EU have not included the NDF in their official list of terrorist groups.
- claim 50: true Conflicting Evidence/Cherrypicking; TF-IDF Refuted; SBERT Refuted; reason conflicting_confusion. Twitter now putting warnings on tweets that are just unedited clips of Biden saying he would end fracking
- claim 115: true Conflicting Evidence/Cherrypicking; TF-IDF Refuted; SBERT Refuted; reason conflicting_confusion. More people who wear masks become sick with COVID-19 compared to non-mask wearers
- claim 148: true Conflicting Evidence/Cherrypicking; TF-IDF Refuted; SBERT Supported; reason conflicting_confusion. U.S. President Donald Trump has suspended all travel from China.
- claim 170: true Supported; TF-IDF Refuted; SBERT Refuted; reason other. U.S. President Donald Trump has departed Walter Reed hospital just a few days after testing positive for COVID-19.

### sbert_lr

- claim 50: true Conflicting Evidence/Cherrypicking; TF-IDF Refuted; SBERT Refuted; reason conflicting_confusion. Twitter now putting warnings on tweets that are just unedited clips of Biden saying he would end fracking
- claim 53: true Supported; TF-IDF Refuted; SBERT Refuted; reason thin_evidence. The FBI is in possession of the president of the US's son Hunter Biden's laptop.
- claim 64: true Refuted; TF-IDF Refuted; SBERT Supported; reason other. Eric Trump, Donald J. Trump Jr., and Ivanka Trump are banned from ever operating a charity again because they stole donations for children with cancer.
- claim 70: true Refuted; TF-IDF Refuted; SBERT Not Enough Evidence; reason other. Dr Anthony Fauci wrote a paper regarding the Spanish Flu and stated that the majority of deaths in 1918-1919 was because of bacterial pneumonia from wearing masks.
- claim 115: true Conflicting Evidence/Cherrypicking; TF-IDF Refuted; SBERT Refuted; reason conflicting_confusion. More people who wear masks become sick with COVID-19 compared to non-mask wearers

## Disagreements

- claim 2 (sbert wins): true Refuted; TF-IDF Not Enough Evidence; SBERT Refuted. Due to Imran Khan's criticism of Macron's comments on Islam, French authorities cancelled the visas of 183 Pakistani citizens and deported 118 from the country.
- claim 32 (sbert wins): true Supported; TF-IDF Refuted; SBERT Supported. The New York post Endorsed Donald trumps bid for re-election.
- claim 93 (sbert wins): true Supported; TF-IDF Refuted; SBERT Supported. US Judge Amy Coney Barrett graduated at the top of her law school class at Notre Dame Law School
- claim 100 (sbert wins): true Conflicting Evidence/Cherrypicking; TF-IDF Refuted; SBERT Conflicting Evidence/Cherrypicking. US president Joe Biden (candidate at the time of claim) supports the new green deal (The Green New Deal is a nonbinding resolution, introduced by Rep. Alexandri...
- claim 181 (sbert wins): true Supported; TF-IDF Refuted; SBERT Supported. The ex-Prime Minister of India Lal Bahadur Shastri took a loan to buy a car.
- claim 297 (sbert wins): true Supported; TF-IDF Refuted; SBERT Supported. Sightway Capital is owned by Two Sigma Investments.
- claim 374 (sbert wins): true Refuted; TF-IDF Supported; SBERT Refuted. $400,000 ... is more money than I’ve ever made” in a year
- claim 412 (sbert wins): true Conflicting Evidence/Cherrypicking; TF-IDF Supported; SBERT Conflicting Evidence/Cherrypicking. Sen. Martha McSally “supported a plan to turn Medicare into a voucher program and ‘shift costs’ onto seniors” and “even supported raising the retirement age.
- claim 449 (sbert wins): true Supported; TF-IDF Refuted; SBERT Supported. All India Institute of Medical Sciences Nursing Officer Recruitment Common Eligibility Test 2020 was postponed.
- claim 482 (sbert wins): true Refuted; TF-IDF Conflicting Evidence/Cherrypicking; SBERT Refuted. This Democratic Party of Joe Biden is pushing the so-called Green New Deal. If given power, they would essentially ban animal agriculture and eliminate gas-powe...
- claim 64 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Supported. Eric Trump, Donald J. Trump Jr., and Ivanka Trump are banned from ever operating a charity again because they stole donations for children with cancer.
- claim 131 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Supported. We do not know where Covid-19 places among causes of death because the data is not published.
- claim 155 (tfidf wins): true Supported; TF-IDF Supported; SBERT Refuted. Trump said 'there were fine people on both sides' in far-right protests.
- claim 164 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Not Enough Evidence. OPCW scenario of the Navalny case is a fantastic story and a planned conspiracy scenario
- claim 193 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Supported. Two E-6B Mercury planes were deployed because of US President Trump's positive COVID-19 test.
- claim 206 (tfidf wins): true Not Enough Evidence; TF-IDF Not Enough Evidence; SBERT Supported. There was a 40% drop in oil prices in Nigeria.
- claim 246 (tfidf wins): true Supported; TF-IDF Supported; SBERT Conflicting Evidence/Cherrypicking. A Maryland man was sentenced to a year in jail for throwing parties.
- claim 260 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Conflicting Evidence/Cherrypicking. False Facebook posts claim Philippine vice president demanded multi-billion dollar budget for her office in 2021
- claim 420 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Supported. US President Trump issued an executive order for the White House to take over all electrical grids, which will include internet servers, broadcasting systems an...
- claim 460 (tfidf wins): true Refuted; TF-IDF Refuted; SBERT Supported. Donald Trump says Biden "is even talking about taking the wall down."
