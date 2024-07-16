#!/usr/bin/env python3
from eval.retrieval.get_articles import main as get_articles_main
from eval.retrieval.get_questions import main as get_questions_main

def main():
    get_articles_main()
    get_questions_main()
    # Set up evals: retrieval (wikipedia) and end2end
    #   (We can put these on huggingface too)
    # Run the evals
    for eval_name in ['retrieval', 'end2end']:
        for benchmark in ['knn', 'dartboard']:
            # for params in []:
            pass

    # Make the plots



if __name__ == "__main__": main()

