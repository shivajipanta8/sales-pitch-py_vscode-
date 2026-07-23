"""Simple sales analysis script for students.

This script reads sales data from a CSV file, cleans the data,
calculates basic summaries, prints a simple report, and saves a few charts.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_FILE = Path("Sales_Data.csv")
PLOT_DIR = Path("plots")


def load_data(file_path: Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path, low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns and extract the month from the purchase date."""
    numeric_columns = ["Total Price", "Final Price", "Discount Amount", "Quantity"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "Purchase Date" in df.columns:
        df["Purchase Date"] = pd.to_datetime(df["Purchase Date"], errors="coerce")
        df["Month"] = df["Purchase Date"].dt.month_name()
    else:
        df["Month"] = None

    return df


def summarize_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Combine rows by order so each order has one summary row."""
    if "Order ID" not in df.columns:
        raise KeyError("The data must contain an 'Order ID' column.")

    summary = df.groupby("Order ID", dropna=False).agg(
        {
            "Customer ID": "first",
            "Customer Type": "first",
            "Store Location": "first",
            "Purchase Date": "first",
            "Month": "first",
            "Total Price": "sum",
            "Final Price": "sum",
            "Discount Amount": "sum",
            "Quantity": "sum",
        }
    )
    summary = summary.reset_index()
    summary.columns = [
        "Order ID",
        "Customer ID",
        "Customer Type",
        "Store Location",
        "Purchase Date",
        "Month",
        "Total Price",
        "Final Price",
        "Discount Amount",
        "Quantity",
    ]
    return summary


def customer_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize revenue and order count for each customer type."""
    return (
        orders.groupby("Customer Type", dropna=False)
        .agg(
            TotalRevenue=("Final Price", "sum"),
            AverageOrderValue=("Final Price", "mean"),
            OrderCount=("Order ID", "nunique"),
            TotalQuantity=("Quantity", "sum"),
        )
        .sort_values("TotalRevenue", ascending=False)
    )


def store_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize how much each store location sold."""
    return (
        orders.groupby("Store Location", dropna=False)
        .agg(
            TotalRevenue=("Final Price", "sum"),
            AverageOrderValue=("Final Price", "mean"),
            OrderCount=("Order ID", "nunique"),
            TotalQuantity=("Quantity", "sum"),
            TotalDiscount=("Discount Amount", "sum"),
        )
        .sort_values("TotalRevenue", ascending=False)
    )


def monthly_summary(orders: pd.DataFrame) -> pd.DataFrame:
    """Summarize revenue and order count for each month."""
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    summary = (
        orders.groupby("Month", dropna=False)
        .agg(
            TotalRevenue=("Final Price", "sum"),
            OrderCount=("Order ID", "nunique"),
            TotalQuantity=("Quantity", "sum"),
        )
        .reindex(months)
        .fillna(0)
    )
    summary["RevenueChange"] = summary["TotalRevenue"].diff()
    return summary


def print_report(
    raw_data: pd.DataFrame,
    orders: pd.DataFrame,
    monthly: pd.DataFrame,
    customers: pd.DataFrame,
    stores: pd.DataFrame,
) -> None:
    print("\n=== Simple Sales Analysis Report ===")
    print(f"Rows in raw data: {len(raw_data)}")
    print(f"Columns: {list(raw_data.columns)}")
    print(f"Missing values: {raw_data.isna().sum().sum()}")

    print("\n--- Revenue Summary ---")
    print(f"Total revenue: ${orders['Final Price'].sum():,.2f}")
    print(f"Average order value: ${orders['Final Price'].mean():,.2f}")
    print(f"Total discount amount: ${orders['Discount Amount'].sum():,.2f}")

    print("\n--- Top 5 Orders by Final Price ---")
    print(orders.nlargest(5, "Final Price")[
        ["Order ID", "Customer ID", "Store Location", "Final Price"]
    ])

    print("\n--- Revenue by Customer Type ---")
    print(customers)

    print("\n--- Revenue by Store Location ---")
    print(stores)

    print("\n--- Monthly Revenue ---")
    print(monthly[["TotalRevenue", "OrderCount"]])


def make_plots(
    orders: pd.DataFrame,
    monthly: pd.DataFrame,
    customers: pd.DataFrame,
    stores: pd.DataFrame,
) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    stores["TotalRevenue"].plot(kind="bar", title="Revenue by Store", figsize=(8, 5))
    plt.xlabel("Store Location")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "revenue_by_store.png")
    plt.close()

    customers["TotalRevenue"].plot(kind="bar", title="Revenue by Customer Type", figsize=(8, 5))
    plt.xlabel("Customer Type")
    plt.ylabel("Revenue")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "revenue_by_customer_type.png")
    plt.close()

    monthly["TotalRevenue"].plot(kind="line", marker="o", title="Monthly Revenue", figsize=(8, 5))
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "monthly_revenue.png")
    plt.close()

    print(f"Saved charts to: {PLOT_DIR}")


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = load_data(DATA_FILE)
    df = clean_data(df)
    orders = summarize_orders(df)
    customers = customer_summary(orders)
    stores = store_summary(orders)
    monthly = monthly_summary(orders)

    print_report(df, orders, monthly, customers, stores)
    make_plots(orders, monthly, customers, stores)


if __name__ == "__main__":
    main()
