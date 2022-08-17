### This is only a starter code to run your tests
### Write/edit/extend this code to fully execute your pipeline and test your assignment1.py


import assignment1

X, y = assignment1.load_prepare()
training_accuracy_1, confusion_matrix_1, pipeline = assignment1.build_pipeline_1(X, y)
training_accuracy_2, confusion_matrix_2, pipeline = assignment1.build_pipeline_2(X, y)
training_accuracy_3, confusion_matrix_3, pipeline = assignment1.build_pipeline_final(X, y)

print('model1 traning accuracy', training_accuracy_1)
print('confusion matrix', confusion_matrix_1)
print('model2 traning accuracy', training_accuracy_2)
print('confusion matrix', confusion_matrix_2)
print('final model traning accuracy', training_accuracy_3)
print('confusion matrix', confusion_matrix_3)

predictions = assignment1.apply_pipeline()


