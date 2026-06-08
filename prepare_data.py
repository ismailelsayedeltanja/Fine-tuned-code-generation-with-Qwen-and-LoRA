from sklearn.model_selection import train_test_split
import json
import os
# هنا هنستخدم json.dumps() علشان نحول الـ dictionary إلى JSON text
# Sample dataset — replace with your own examples
# random.shuffle() علشان نلخبط ترتيب البيانات قبل التقسيم
EXAMPLES = [           # دي dataset نفسها
                       # عبارة عن List
                       # كل عنص فيها Dictionary
  '''
1.EXAMPLES
   
2.split_data()
  
3.train_data + val_data
   
4.save_jsonl()
  
5.train.jsonl + val.jsonl

  '''
     {
        "instruction": "Write a Python function that returns the factorial of a number.",
        "input": "",      #Write a Python function
        "output": (
            "def factorial(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    if n == 0:\n"
            "        return 1\n"
            "    result = 1\n"
            "    for i in range(1, n + 1):\n"
            "        result *= i\n"
            "    return result"
        ),
    },
    {
        "instruction": "Write a function that checks if a string is a palindrome.",
        "input": "",    # بيعمل lowercase w حذف المسافات
        "output": (
            "def is_palindrome(s: str) -> bool:\n"
            "    cleaned = s.lower().replace(' ', '')\n"
            "    return cleaned == cleaned[::-1]" # دي slicing لعكس الـ string
        ),
    },
    {
        "instruction": "Add type hints and a docstring to this function.",
        "input": (
            "def divide(a, b):\n"   # قسمة رقمين
            "    return a / b"
        ),
        "output": (
            "def divide(a: float, b: float) -> float:\n"
            "    \"\"\"\n"
            "    Divide a by b.\n\n"
            "    Args:\n"
            "        a: The dividend.\n"
            "        b: The divisor.\n\n"
            "    Returns:\n"
            "        The result of a / b.\n\n"
            "    Raises:\n"
            "        ZeroDivisionError: If b is zero.\n"
            "    \"\"\"\n"
            "    if b == 0:\n"
            "        raise ZeroDivisionError('Cannot divide by zero')\n"
            "    return a / b"
        ),
    },
    {
        "instruction": "Write a Python class for a simple stack data structure.",
        "input": "",   #Write a Python class
        "output": (
            "class Stack:\n"
            "    def __init__(self):\n"
            "        self._items = []\n\n"
            "    def push(self, item):\n"
            "        self._items.append(item)\n\n"
            "    def pop(self):\n"
            "        if self.is_empty():\n"
            "            raise IndexError('Stack is empty')\n"
            "        return self._items.pop()\n\n"
            "    def peek(self):\n"
            "        if self.is_empty():\n"
            "            raise IndexError('Stack is empty')\n"
            "        return self._items[-1]\n\n"
            "    def is_empty(self) -> bool:\n"
            "        return len(self._items) == 0\n\n"
            "    def size(self) -> int:\n"
            "        return len(self._items)"
        ),
    },
    {
        "instruction": "Write a function that reads a JSON file and returns its content as a dictionary.",
        "input": "",     #Write a function that reads a JSON file
        "output": (
            "import json\n\n"
            "def read_json(file_path: str) -> dict:\n"
            "    with open(file_path, 'r', encoding='utf-8') as f:\n"
            "        return json.load(f)"
        ),
    },
    {
        "instruction": "Write a Python function to flatten a nested list.",
        "input": "",    #بيعلّم  model إزاي يكت function بتفك  Nested Lists
        "output": (
            "def flatten(nested: list) -> list:\n"
            "    result = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        ),
    },
    {
        "instruction": "Write a function that counts word frequency in a string.",
        "input": "",      # يعرف كل كلمة اتكررت كام مرة
        "output": (
            "from collections import Counter\n\n"
            "def word_frequency(text: str) -> dict:\n"
            "    words = text.lower().split()\n"
            "    return dict(Counter(words))"
        ),
    },
    {
        "instruction": "Write a Python decorator that measures function execution time.",
        "input": "",   # أي function تحط فوقها timer هيطبع مدة تنفيذها بالثواني
        "output": (
            "import time\nfrom functools import wraps\n\n"
            "def timer(func):\n"
            "    @wraps(func)\n"
            "    def wrapper(*args, **kwargs):\n"
            "        start = time.perf_counter()\n"
            "        result = func(*args, **kwargs)\n"
            "        elapsed = time.perf_counter() - start\n"
            "        print(f'{func.__name__} took {elapsed:.4f} seconds')\n"
            "        return result\n"
            "    return wrapper"
        ),
    },
]




#from sklearn.model_selection import train_test_split
# Split data 
def split_data(examples: list, train_ratio: float = 0.8):

    train_data, val_data = train_test_split(
        examples,
        test_size=0.2, #train_size = train_ratio = 0.8
        shuffle=True,  # test_size = 1 - train_ratio
        random_state=101,
    )

    return train_data, val_data


# Save examples aqs JSONL
def save_jsonl(examples: list, path: str):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:

        for ex in examples:
            f.write(
                json.dumps(ex, ensure_ascii=False) + "\n"
            )

    print(f"Saved {len(examples)} examples to {path}")


if __name__ == "__main__":

    train_data, val_data = split_data(EXAMPLES)

    save_jsonl(train_data, "data/train.jsonl")
    save_jsonl(val_data, "data/val.jsonl")

    print("Data preparation complete.")
    print(f"Train: {len(train_data)} examples")
    print(f"Val:   {len(val_data)} examples")



