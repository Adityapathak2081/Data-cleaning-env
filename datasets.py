import json

DIRTY_DATASETS = [
    {
        "name": "employee_data",
        "data": [
            {"name": "Alice",  "age": 30,    "salary": 50000, "department": "HR"},
            {"name": "Bob",    "age": None,  "salary": 60000, "department": "IT"},
            {"name": "Carol",  "age": 25,    "salary": None,  "department": "hr"},
            {"name": "Bob",    "age": None,  "salary": 60000, "department": "IT"},
            {"name": "Dave",   "age": 999,   "salary": 55000, "department": "Finance"},
            {"name": "",       "age": 28,    "salary": 52000, "department": "IT"},
            {"name": "Eve",    "age": "abc", "salary": 48000, "department": "HR"},
        ],
        "issues": {
            "nulls": ["age", "salary"],
            "duplicates": True,
            "outliers": {"age": (0, 100)},
            "type_errors": {"age": "int"},
            "formatting": {"department": "title_case"}
        }
    }
]

def get_dataset(name="employee_data"):
    """Returns the dirty dataset by name."""
    for d in DIRTY_DATASETS:
        if d["name"] == name:
            return d
    return DIRTY_DATASETS[0]


# Quick test — run this file directly to see the data
if __name__ == "__main__":
    dataset = get_dataset()
    print("Dataset name:", dataset["name"])
    print("Number of rows:", len(dataset["data"]))
    print("\nData:")
    for row in dataset["data"]:
        print(" ", row)
    print("\nKnown issues:", list(dataset["issues"].keys()))