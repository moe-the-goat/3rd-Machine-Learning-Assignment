# 3rd Machine Learning Assignment

Hey, welcome to my repo! This is my third machine learning assignment, and honestly, it was quite a ride putting this together. Let me walk you through what I did here.

## So What Is This Project? 

Basically, we were given a dataset collected from students, and the goal was to build models that can predict country-based classifications.  Sounds simple enough, right? Well, it turned out to be more interesting than I initially thought.  From cleaning messy data to figuring out why my models were making certain mistakes, there was a lot to unpack.

## How I Organized Everything

I tried to keep things neat so I wouldn't lose my mind going back and forth between files. Here's the breakdown:

The **code** folder is where all the Python scripts live. You'll find everything from data exploration to the final error analysis there.

The **output folders** like EDA_Output, Task1_Output, Task3_Baseline, Task4_Models, and Task5_ErrorAnalysis contain all the results, charts, and visualizations I generated along the way.

The **dataset** is that CSV file sitting in the root directory.  That's our raw material. 

And yeah, there's also the assignment PDF in case you want to see what the actual requirements were.

## The Process I Followed

Let me break down how I approached this thing from start to finish. 

### Defining the Learning Task

Before jumping into any code, I had to sit down and really understand what problem I was trying to solve. What's the target variable? What features make sense? What kind of prediction is this, classification or regression? Getting this right early saved me a lot of headaches later.

### Exploring the Data

This is probably where I spent the most time, honestly. I went through the dataset pretty thoroughly, looking at distributions, checking for missing values, spotting weird outliers, and understanding how different features relate to each other.  You really can't skip this step because the data will surprise you in ways you don't expect. 

### Building Baseline Models

I always like to start simple. I trained a K-Nearest Neighbors model and a Logistic Regression model just to get a sense of what kind of performance is achievable. These aren't fancy, but they give you a reference point.  If your complex model can't beat logistic regression, something's probably off. 

### Training Better Models

Once I had my baselines, I moved on to more sophisticated approaches. The idea was to see how much improvement I could squeeze out compared to the simple models.  This is where things got interesting and also a bit frustrating at times. 

### Analyzing Errors

This was actually my favorite part. Instead of just looking at accuracy numbers, I dug into where the models were making mistakes. Which classes were getting confused? Were there patterns in the misclassifications? This kind of analysis tells you so much more than a single metric ever could.

## What You Need to Run This

You'll need Python with the usual data science stack.  Pandas for handling data, NumPy for numerical stuff, Scikit-learn for the machine learning bits, and Matplotlib along with Seaborn for visualizations. If you've done any ML work before, you probably have all of this installed already.

## Running the Code

Just clone the repo, navigate into the folder, and run the main pipeline script. Or if you want to run individual tasks separately, each one has its own Python file in the code folder.  Pretty straightforward. 

## What I Learned

Going through this assignment taught me a few things worth mentioning.  First, never underestimate exploratory data analysis.  It's tempting to jump straight into modeling, but understanding your data first makes everything else smoother. Second, baselines are your friend. They keep you grounded and give you something concrete to improve upon. Third, error analysis deserves more attention than it usually gets. That's where you actually learn what your model is doing wrong and how you might fix it.

## Final Thoughts

This project took some effort, but I'm happy with how it turned out. If you're going through something similar or just curious about the approach I took, feel free to poke around the code and outputs. And if anything's unclear, don't hesitate to reach out. 

Thanks for stopping by!