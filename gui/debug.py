import dataclasses

@dataclasses.dataclass(frozen=True)
class DataClass:
    name: str = 'oi'
    value: int = 1


obj1 = DataClass()
print(obj1.value)

obj1.value = 2
print(obj1.value)

obj2 = DataClass(1, 'ola')
print(obj2.name)
print(obj2.value)

