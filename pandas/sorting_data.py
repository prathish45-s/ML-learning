import pandas as pd

import pandas as pd

df = pd.DataFrame({
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha"],
    "Age": [20, 22, 21, 23, 25],
    "Marks": [75, 82, 68, 90, 88]
})

print(df.sort_values("Marks", ascending=False))