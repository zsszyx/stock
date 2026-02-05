from stock.data_context import ConceptContext

def test_concept_context():
    print("Loading Concept Context...")
    ctx = ConceptContext()
    
    # Test individual stock
    sample_stock = ctx.all_stocks[0]
    concepts = ctx.get_concepts(sample_stock)
    print(f"Stock {sample_stock} has concepts: {concepts}")
    
    # Test individual concept
    sample_concept = ctx.all_concepts[0]
    stocks = ctx.get_stocks(sample_concept)
    print(f"Concept '{sample_concept}' has {len(stocks)} stocks.")
    
    # Test common concepts for a group
    # Let's take first 5 stocks
    group = ctx.all_stocks[:5]
    common = ctx.get_common_concepts(group)
    print(f"Common concepts for {group}:")
    for cp, count in list(common.items())[:5]:
        print(f"  - {cp}: {count}")

if __name__ == "__main__":
    test_concept_context()
