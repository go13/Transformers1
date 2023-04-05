from ga.ga import GA, TargetStringEvaluator

ga = GA(TargetStringEvaluator())

for _ in range(10000):
    ga.step()
