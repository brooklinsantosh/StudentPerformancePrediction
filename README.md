# StudentPerformancePrediction

### Software and Tools Requirements

1. [Github](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com)
3. [Heroku](https://heroku.com)
4. [GitCLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

### Create a new virtual environment
```
conda create -p venv python=3.12 -y
```

### Installing all the dependencies
```
pip install -r requirements.txt
```

### Problem statement
- This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.


### Data Collection
- Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
- The data consists of 8 column and 1000 rows.

#### Data description
- gender : sex of students  -> (Male/female)
- race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
- parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
- lunch : having lunch before test (standard or free/reduced) 
- test preparation course : complete or not complete before test
- math score
- reading score
- writing score