
import argparse
import json
import typing as t
from dataclasses import dataclass, asdict
from pathlib import Path

import dataset




@dataclass
class FictionalBrand:
    brand: str  
    model: str  
    brand_code: str  
    model_code: str  
    is_fictional: bool = True  
    knowledge_strength: float = 0.0  


@dataclass
class CategoryFictionalBrands:
    category: str
    fictional_brands: t.List[FictionalBrand]





FICTIONAL_BRAND_TEMPLATES = [
    {"brand_code": "X", "model_code": "X1"},
    {"brand_code": "Y", "model_code": "Y1"},
    {"brand_code": "Z", "model_code": "Z1"},
    {"brand_code": "W", "model_code": "W1"},
]


def create_fictional_brand(brand_code: str, model_code: str) -> FictionalBrand:
    return FictionalBrand(
        brand=f"Brand_{brand_code}",
        model=f"Model_{model_code}",
        brand_code=brand_code,
        model_code=model_code,
        is_fictional=True,
        knowledge_strength=0.0
    )


def create_all_fictional_brands() -> t.List[FictionalBrand]:
    return [
        create_fictional_brand(template["brand_code"], template["model_code"])
        for template in FICTIONAL_BRAND_TEMPLATES
    ]


def create_fictional_brands_for_category(category: str) -> CategoryFictionalBrands:
    fictional_brands = create_all_fictional_brands()
    return CategoryFictionalBrands(
        category=category,
        fictional_brands=fictional_brands
    )


def create_fictional_brands_for_all_categories() -> t.Dict[str, CategoryFictionalBrands]:
    categories = dataset.get_categories()
    results = {}
    
    for category in categories:
        results[category] = create_fictional_brands_for_category(category)
    
    return results




def save_fictional_brands(
    results: t.Dict[str, CategoryFictionalBrands],
    output_dir: str = "./out/fictional_brands"
):
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    serializable_results = {}
    for category, category_brands in results.items():
        serializable_results[category] = {
            "category": category_brands.category,
            "fictional_brands": [asdict(fb) for fb in category_brands.fictional_brands]
        }
    
    
    output_file = save_dir / "fictional_brands.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Fictional brands saved to: {output_file}")
    
    
    summary_file = save_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("Fictional brand list (not present in LLM parametric knowledge)\n")
        f.write("=" * 50 + "\n\n")
        
        fictional_brands = create_all_fictional_brands()
        for fb in fictional_brands:
            f.write(f"- {fb.brand} || {fb.model}\n")
        
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Total fictional brands: {len(fictional_brands)}\n")
        f.write("These brands are used to evaluate whether the LLM is biased toward brands in its parametric knowledge.\n")
    
    print(f"Summary saved to: {summary_file}")


def load_fictional_brands(
    output_dir: str = "./out/fictional_brands"
) -> t.Dict[str, CategoryFictionalBrands]:
    input_file = Path(output_dir) / "fictional_brands.json"
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    results = {}
    for category, category_data in data.items():
        fictional_brands = [
            FictionalBrand(**fb_data)
            for fb_data in category_data["fictional_brands"]
        ]
        results[category] = CategoryFictionalBrands(
            category=category,
            fictional_brands=fictional_brands
        )
    
    return results




def get_fictional_brands() -> t.List[FictionalBrand]:
    return create_all_fictional_brands()


def get_fictional_brands_as_tuples() -> t.List[t.Tuple[str, str]]:
    return [
        (fb.brand, fb.model)
        for fb in create_all_fictional_brands()
    ]




def print_fictional_brands():
    print("\n" + "=" * 60)
    print("Fictional brand list (not present in LLM parametric knowledge)")
    print("=" * 60)
    
    fictional_brands = create_all_fictional_brands()
    
    for i, fb in enumerate(fictional_brands, 1):
        print(f"\n{i}. {fb.brand} || {fb.model}")
        print(f"   Codes: ({fb.brand_code}, {fb.model_code})")
        print(f"   Knowledge strength: {fb.knowledge_strength} (fictional brand)")
    
    print("\n" + "=" * 60)
    print(f"Total fictional brands: {len(fictional_brands)}")
    print("=" * 60)




def main():
    parser = argparse.ArgumentParser(
        description="Create fictional brands that are not present in LLM parametric knowledge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="./out/fictional_brands",
        help="Output directory"
    )
    parser.add_argument(
        "--test", type=str, default=None,
        help="Test a single category, e.g. 'laptop'"
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Only print fictional brands; do not save files"
    )
    
    args = parser.parse_args()
    
    if args.print_only:
        print_fictional_brands()
        return
    
    if args.test:
        
        print(f"\n[Test mode] Category: {args.test}")
        result = create_fictional_brands_for_category(args.test)
        
        print(f"\nFictional brands for category '{args.test}':")
        for fb in result.fictional_brands:
            print(f"  - {fb.brand} || {fb.model}")
    else:
        
        print("\nCreating fictional brands for all categories...")
        results = create_fictional_brands_for_all_categories()
        
        print(f"\nFound {len(results)} product categories")
        
        
        save_fictional_brands(results, args.output_dir)
        
        
        print_fictional_brands()


if __name__ == "__main__":
    main()
