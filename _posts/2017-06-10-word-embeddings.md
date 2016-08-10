---
layout: post
comments: true
title:  "Word Embeddings: What, Why and How"
date:   2017-06-11 22:00:00
categories: deep learning
---

WHEN IN DOUBT, START FROM SCRATCH, START WITH FIRST PRINCIPLES.

1. The Google's ```word2vec``` tool - https://code.google.com/archive/p/word2vec/ and https://github.com/dav/word2vec

2. Build and test:

```
git clone https://github.com/dav/word2vec.git
cd word2vec/
curl http://mattmahoney.net/dc/text8.zip > ../data/text8.gz
(unzip it)
../bin/word2vec -train ../data/text8 -output ../data/text8-vector.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 1
sh demo-phrases.sh
ls -lt ../data
total 764880
-rw-r--r--  1 jefftang  _softwareupdate  101776196 Jun 12 14:00 vectors-phrase.bin
-rw-r--r--  1 jefftang  _softwareupdate   99999948 Jun 12 13:57 text8-phrases
-rw-r--r--  1 jefftang  _softwareupdate   57704809 Jun 12 13:44 text8-vector.bin
-rw-r--r--  1 jefftang  _softwareupdate   31344016 Jun 12 13:38 text8.gz
-rw-r--r--  1 jefftang  _softwareupdate     168209 Jun 12 12:56 questions-phrases.txt
-rw-r--r--  1 jefftang  _softwareupdate     603955 Jun 12 12:56 questions-words.txt
-rwxr-xr-x  1 jefftang  _softwareupdate  100000000 Jun  9  2006 text8

../bin/distance  ../data/vectors-phrase.bin
Enter word or sentence (EXIT to break): love

Word: love  Position in vocabulary: 601

                                              Word       Cosine distance
------------------------------------------------------------------------
                                                me		0.577848
                                                my		0.575793
                                    porter_wagoner		0.545939
                                              girl		0.498306
                                             loves		0.495461
                                            hailie		0.491285
                                             songs		0.487081
                                             gonna		0.483994
                                              song		0.477809
                                           tonight		0.475846
                                              your		0.470771
                                              thee		0.464838
                                           goodbye		0.464676
                                            loving		0.463634
                                            hooray		0.461145
                                             kylie		0.458195
                                           souffle		0.455974
                                     lose_yourself		0.453592
                                               you		0.448788
                                              dear		0.448167
                                           grendel		0.447531
                                              baby		0.446445
                                            my_dad		0.443778
                                             smile		0.440657
                                           connick		0.439675
                                          jim_waco		0.437465
                                    red_ridinghood		0.437153
                                   neighbor_totoro		0.432810
                                             crazy		0.431641
                                         everybody		0.430092
                                            madman		0.427392
                                            punkie		0.426429
                                             ain_t		0.426057
                                          liaisons		0.425851
                                               man		0.425822
                                             don_t		0.423506
                                             woman		0.422720
                                           elektra		0.420637
                                             nigga		0.418828
                                            mother		0.418812
Enter word or sentence (EXIT to break): dog

Word: dog  Position in vocabulary: 1901

                                              Word       Cosine distance
------------------------------------------------------------------------
                                              dogs		0.626967
                                          keeshond		0.617924
                                  belgian_shepherd		0.605499
                                             breed		0.515554
                                          komondor		0.502236
                                              cuon		0.500997
                                           terrier		0.499316
                                               cat		0.492947
                                             jindo		0.489221
                                             hound		0.481776
                                           mastiff		0.470266
                                          hairless		0.461524
                                              cats		0.455848
                                           leopard		0.450873
                                             canis		0.449708
                                               aye		0.449528
                                              puma		0.443623
                                          sheepdog		0.443162
                                     munsterlander		0.441639
                                      ibizan_hound		0.441515
                                        sighthound		0.440437
                                        keeshonden		0.437598
                                        wirehaired		0.436333
                                            borzoi		0.431837
                                       canis_lupus		0.430988
                                               rat		0.423408
                                            ruffed		0.420480
                                    asked_zhaozhou		0.420391
                                            donkey		0.420253
                                              wolf		0.420078
                                      shepherd_dog		0.418551
                                 italian_greyhound		0.418293
                                           spaniel		0.416497
                                           canidae		0.416248
                                        familiaris		0.414550
                                          pyrenean		0.414151
                                   russell_terrier		0.413815
                                        dog_breeds		0.413485
                                               ass		0.411225
                                            canine		0.411076
Enter word or sentence (EXIT to break): china

Word: china  Position in vocabulary: 465

                                              Word       Cosine distance
------------------------------------------------------------------------
                                            taiwan		0.708759
                                          republic		0.609633
                                             korea		0.599966
                                    mainland_china		0.584838
                                               prc		0.581067
                                               roc		0.567405
                                         hong_kong		0.560530
                                          shanghai		0.554945
                                       south_korea		0.544441
                                             macau		0.535586
                                         singapore		0.534897
                                           nanjing		0.533253
                                             japan		0.529232
                                           chinese		0.526099
                                            taipei		0.524640
                                             tibet		0.523735
                                           beijing		0.520297
                                           beiping		0.518761
                                          thailand		0.504753
                                          mongolia		0.491219
                                          malaysia		0.490499
                                        chiang_kai		0.490073
                                           vietnam		0.488878
                                             macao		0.488827
                                              laos		0.486459
                                          cambodia		0.485985
                                            chiang		0.484380
                                              shek		0.478316
                                      cuon_alpinus		0.474374
                                           sun_yat		0.467555
                                        tian_anmen		0.465184
                                      chairpersons		0.461694
                                 mahayana_buddhism		0.459855
                                          xinjiang		0.458970
                                             india		0.455404
                                  korean_peninsula		0.452589
                                       shimonoseki		0.450883
                              diplomatic_relations		0.446197
                                             fyrom		0.444893
                                    southeast_asia		0.444293
Enter word or sentence (EXIT to break):
```

3. App ideas:
 - visualization
 - interaction

4. References:
- [Representing Words](http://veredshwartz.blogspot.co.il/2016/01/representing-words.html)
- [word2vec in Java](http://deeplearning4j.org/word2vec.html)
- [Word2Vec Introduction in Python - example not complete](http://www.folgertkarsdorp.nl/word2vec-an-introduction/)
  - complete the example; fully understand "the posterior probability of an output word given some input word"...
- [word2vec Parameter Learning Explained](http://arxiv.org/pdf/1411.2738v4.pdf)
- [Distributed Representations of Sentences and Documents](https://github.com/fbkarsdorp/doc2vec/blob/master/doc2vec.ipynb)

- [Nice example on word2vec](https://iksinc.wordpress.com/tag/continuous-bag-of-words-cbow/)

TODO: implement word 2 vec from scratch, using one word continuous-bag-of-words.


   * representing words
   * representing animals
   * representing people
   * representing movies, music, songs, items, objects..
