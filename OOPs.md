Got it ✅
Here’s the **full 1–100 Python OOPs Interview Questions & Answers** in **Markdown format** so you can directly copy-paste into a `.md` file for GitHub.

````markdown
# Python OOPs Interview Questions & Answers

---

### Q1. What is OOP?
Object-Oriented Programming (OOP) is a programming paradigm based on the concept of objects, which contain data (attributes) and methods (functions).

---

### Q2. What are the main principles of OOP?
- Encapsulation  
- Inheritance  
- Polymorphism  
- Abstraction  

---

### Q3. What is a class in Python?
A class is a blueprint for creating objects, defining attributes and methods.

```python
class Car:
    def __init__(self, brand):
        self.brand = brand
````

---

### Q4. What is an object in Python?

An object is an instance of a class.

```python
car1 = Car("Tesla")
```

---

### Q5. What is `__init__` in Python?

`__init__` is a constructor method used to initialize object attributes.

---

### Q6. What is the difference between class and object?

* Class: Blueprint
* Object: Instance of a class

---

### Q7. What is encapsulation?

Encapsulation is bundling data and methods inside a class and restricting access to them.

---

### Q8. How is encapsulation implemented in Python?

Using public, protected, and private members.

```python
class A:
    def __init__(self):
        self.public = 1
        self._protected = 2
        self.__private = 3
```

---

### Q9. What is inheritance?

Inheritance allows one class to acquire properties and methods of another.

---

### Q10. What are the types of inheritance in Python?

* Single
* Multiple
* Multilevel
* Hierarchical
* Hybrid

---

### Q11. Give an example of single inheritance.

```python
class Parent:
    def greet(self):
        print("Hello")

class Child(Parent):
    pass
```

---

### Q12. Give an example of multiple inheritance.

```python
class A:
    def methodA(self):
        print("A")

class B:
    def methodB(self):
        print("B")

class C(A, B):
    pass
```

---

### Q13. What is multilevel inheritance?

When a class inherits from a derived class.

---

### Q14. What is hierarchical inheritance?

Multiple child classes inherit from a single parent class.

---

### Q15. What is hybrid inheritance?

Combination of multiple types of inheritance.

---

### Q16. What is polymorphism?

Ability of objects to take multiple forms (method overloading, overriding).

---

### Q17. What is method overloading?

Same method name with different parameter lists (not natively supported in Python, but can be mimicked).

---

### Q18. What is method overriding?

Child class provides a new implementation of a method inherited from parent.

---

### Q19. Example of method overriding.

```python
class Parent:
    def greet(self):
        print("Hello Parent")

class Child(Parent):
    def greet(self):
        print("Hello Child")
```

---

### Q20. What is abstraction?

Hiding implementation details and showing only essential features.

---

### Q21. How to implement abstraction in Python?

Using `abc` module and abstract base classes.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
```

---

### Q22. What is the difference between abstraction and encapsulation?

* Abstraction: Hides implementation details.
* Encapsulation: Restricts access to data.

---

### Q23. What are access specifiers in Python?

* Public: `var`
* Protected: `_var`
* Private: `__var`

---

### Q24. Is multiple inheritance supported in Python?

Yes, Python supports multiple inheritance.

---

### Q25. What is MRO (Method Resolution Order)?

Order in which methods are searched in multiple inheritance. Use `ClassName.mro()`.

---

### Q26. What is `super()` in Python?

`super()` is used to call parent class methods.

---

### Q27. Example of `super()`.

```python
class Parent:
    def __init__(self):
        print("Parent")

class Child(Parent):
    def __init__(self):
        super().__init__()
        print("Child")
```

---

### Q28. What is `self` in Python classes?

`self` refers to the instance of the class.

---

### Q29. Difference between class method and static method?

* Class method: Has access to class variables (`@classmethod`).
* Static method: No access to class/instance variables (`@staticmethod`).

---

### Q30. Example of class method.

```python
class A:
    count = 0
    @classmethod
    def increment(cls):
        cls.count += 1
```

---

### Q31. Example of static method.

```python
class Math:
    @staticmethod
    def add(x, y):
        return x + y
```

---

### Q32. What is a destructor in Python?

`__del__` method, called when an object is destroyed.

---

### Q33. Example of destructor.

```python
class A:
    def __del__(self):
        print("Destroyed")
```

---

### Q34. What are dunder methods?

Special methods in Python starting with `__` and ending with `__`.

---

### Q35. Example of dunder methods.

* `__init__`
* `__str__`
* `__len__`
* `__add__`

---

### Q36. What is operator overloading?

Defining custom behavior for operators using dunder methods.

---

### Q37. Example of operator overloading.

```python
class Point:
    def __init__(self, x):
        self.x = x
    def __add__(self, other):
        return self.x + other.x
```

---

### Q38. What is `__str__` method?

Defines how the object is represented as a string.

---

### Q39. Example of `__str__`.

```python
class A:
    def __str__(self):
        return "Object of A"
```

---

### Q40. What is `__repr__` method?

Defines unambiguous string representation of object.

---

### Q41. Difference between `__str__` and `__repr__`?

* `__str__`: User-friendly
* `__repr__`: Developer-friendly

---

### Q42. What is the use of `isinstance()`?

Checks if an object is instance of a class.

---

### Q43. What is the use of `issubclass()`?

Checks if a class is a subclass of another.

---

### Q44. What are abstract base classes (ABC)?

Classes that cannot be instantiated and enforce method implementation in subclasses.

---

### Q45. Can Python have private constructors?

Not directly, but can be simulated using private methods.

---

### Q46. What is duck typing in Python?

Python’s dynamic typing style where type is not checked, only presence of methods.

---

### Q47. Example of duck typing.

```python
class Duck:
    def quack(self): print("Quack")

class Person:
    def quack(self): print("I quack like a duck")

def make_it_quack(obj):
    obj.quack()
```

---

### Q48. What is composition in OOP?

Using objects of other classes inside a class.

---

### Q49. Example of composition.

```python
class Engine: pass
class Car:
    def __init__(self):
        self.engine = Engine()
```

---

### Q50. What is aggregation?

A weak form of association where objects can exist independently.

---

### Q51. Difference between composition and aggregation?

* Composition: Strong ownership, lifetime tied.
* Aggregation: Weak ownership, independent lifetime.

---

### Q52. What are class variables?

Variables shared by all instances of a class.

---

### Q53. What are instance variables?

Variables unique to each instance.

---

### Q54. What is multiple dispatch?

Python does not support it directly; can be mimicked with `functools.singledispatch`.

---

### Q55. What is mixin in Python?

A small class designed to add functionality to other classes through inheritance.

---

### Q56. Example of mixin.

```python
class LoggerMixin:
    def log(self, msg):
        print("Log:", msg)
```

---

### Q57. What is metaclass in Python?

Class of a class; defines how classes behave.

---

### Q58. Example of metaclass.

```python
class Meta(type):
    def __new__(cls, name, bases, dct):
        return super().__new__(cls, name, bases, dct)
```

---

### Q59. What is `type()` in Python?

`type()` returns the class type of an object or creates a new class.

---

### Q60. Difference between `is` and `==`?

* `is`: Identity
* `==`: Value equality

---

### Q61. What are slots in Python classes?

`__slots__` restricts object attributes and saves memory.

---

### Q62. Example of `__slots__`.

```python
class A:
    __slots__ = ['x', 'y']
```

---

### Q63. What is property decorator in Python?

Used to define getters/setters in Pythonic way.

---

### Q64. Example of property.

```python
class A:
    def __init__(self, x):
        self._x = x
    @property
    def x(self):
        return self._x
```

---

### Q65. Difference between `@classmethod` and `@staticmethod`?

* `classmethod`: Works with class-level data.
* `staticmethod`: Independent utility method.

---

### Q66. Can Python have abstract properties?

Yes, using `@property` with `abc.ABC`.

---

### Q67. What is method resolution order (MRO)?

It defines the order in which base classes are searched for methods.

---

### Q68. How to check MRO?

```python
print(ClassName.mro())
```

---

### Q69. What is diamond problem in Python OOP?

Ambiguity in multiple inheritance. Python solves it using C3 linearization.

---

### Q70. What is object introspection?

Examining object’s attributes/methods at runtime (`dir()`, `type()`).

---

### Q71. What is monkey patching?

Dynamically modifying classes/objects at runtime.

---

### Q72. Example of monkey patching.

```python
class A: pass
def new_method(self): print("Patched")
A.method = new_method
```

---

### Q73. What is dynamic method binding?

Method resolution occurs at runtime in Python.

---

### Q74. What is static binding?

When method resolution occurs at compile time (not in Python).

---

### Q75. What is overloading vs overriding?

* Overloading: Same name, different params (not native in Python).
* Overriding: Redefining inherited method.

---

### Q76. What is interface in OOP?

A contract that classes must implement (Python uses ABC for this).

---

### Q77. How does Python support multiple interfaces?

By allowing multiple inheritance with abstract base classes.

---

### Q78. What is cooperative multiple inheritance?

Using `super()` ensures proper order in multiple inheritance.

---

### Q79. What is class decorator?

Decorator applied to a class to modify its behavior.

---

### Q80. Example of class decorator.

```python
def decorate(cls):
    cls.new_attr = 100
    return cls

@decorate
class A: pass
```

---

### Q81. What is singleton pattern?

Design pattern where only one instance of a class exists.

---

### Q82. Example of singleton.

```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

---

### Q83. What is factory pattern?

Pattern that creates objects without exposing creation logic.

---

### Q84. Example of factory.

```python
class ShapeFactory:
    def get_shape(self, type):
        if type == "circle": return Circle()
```

---

### Q85. What is observer pattern?

Pattern where objects get notified of state changes in another object.

---

### Q86. What is decorator pattern?

Wrapping objects to add functionality dynamically.

---

### Q87. What is adapter pattern?

Allows incompatible interfaces to work together.

---

### Q88. What is proxy pattern?

A class functioning as an interface to another class.

---

### Q89. What is `__call__` method?

Makes an object callable like a function.

```python
class A:
    def __call__(self):
        print("Called")
```

---

### Q90. What is `__new__` method?

Responsible for creating a new instance before `__init__`.

---

### Q91. Difference between `__new__` and `__init__`?

* `__new__`: Creates object.
* `__init__`: Initializes object.

---

### Q92. What is class inheritance order in Python?

Defined by C3 linearization algorithm.

---

### Q93. Can Python support operator overloading?

Yes, using dunder methods like `__add__`, `__mul__`.

---

### Q94. What is runtime polymorphism?

Polymorphism achieved at runtime via method overriding.

---

### Q95. What is compile-time polymorphism?

Polymorphism resolved at compile time (not applicable in Python).

---

### Q96. What is object cloning in Python?

Creating a copy using `copy` module.

---

### Q97. What is shallow copy vs deep copy?

* Shallow: Copies references.
* Deep: Copies objects recursively.

---

### Q98. Example of shallow copy.

```python
import copy
a = [1, [2]]
b = copy.copy(a)
```

---

### Q99. Example of deep copy.

```python
import copy
a = [1, [2]]
b = copy.deepcopy(a)
```

---

### Q100. Summarize OOP in Python.

OOP in Python enables encapsulation, inheritance, polymorphism, and abstraction to build reusable, modular, and maintainable code.

---

```

✅ This is the **full 1–100 Python OOP Q&A** in **Markdown format**.  

Do you want me to move on to **Python Data Structures (Lists, Dicts, Sets, Tuples) 1–100 Q&A in Markdown** next?
```
