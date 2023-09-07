import random
import csv


def generate_data(sutudentCount, seed=None, fileSave=False):
    subjects = ["Mathematics", "English",
                "Data Structures and Algorithms", "High Performance Computing"]
    students = sutudentCount
    grades = []

    if (seed):
        random.seed(seed)

    for _ in range(students):
        student_grades = {}
        for subject in subjects:
            grade = random.randint(0, 100)
            student_grades[subject] = grade
        grades.append(student_grades)

    if (fileSave):
        # Write grades to a CSV file
        filename = "student_grades.csv"
        with open(filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["Student"] + subjects)
            writer.writeheader()
            for student, grades in enumerate(grades, start=1):
                row = {"Student": f"Student {student}"}
                row.update(grades)
                writer.writerow(row)

    return grades
