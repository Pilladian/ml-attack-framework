# Lookup Table

## AT&T

File format: \<subject\>\_\<glasses\>\_\<img_id\>.png

| subject | meaning
|---      |---
| 0       | subject 0
| ...     | ...
| 39      | subject 39

| glasses | meaning
|---      |---
| 0       | does not wear glasses
| 1       | wears glasses

| img_id  | meaning
|---      |---
| 0       | 0th image of this subject
| ...     | ...
| 10      | 10th image of this subject

## CIFAR10

File format: \<label\>\_\<rand_int\>.png

| label | meaning
|---    |---
| 0     | airplane
| 1     | automobile
| 2     | bird
| 3     | cat
| 4     | deer
| 5     | dog
| 6     | frog
| 7     | horse
| 8     | ship
| 9     | truck

---

## UTKFace

File format: \<age\>\_\<gender\>\_\<race\>\_\<date\>.jpg

| age    | meaning
|---     |---
| 0      | 0 years old
| ...    | ...
| 116    | 116 years old

| gender | meaning
|---     |---
| 0      | male
| 1      | female

| race   | meaning
|---     |---
| 0      | White
| 1      | Black
| 2      | Asian
| 3      | Indian
| 4      | Others
