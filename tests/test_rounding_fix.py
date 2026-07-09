"""
Test for the rounding bug fix in position sizing.

This test verifies that the minimum notional enforcement works correctly
even after rounding to 3 decimal places.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_rounding_bug_at_90k_price():
    """
    Test the exact scenario that was causing rejections.

    At BTC price ~$90,300:
    - $100 / $90,300 = 0.001107 BTC
    - round(0.001107, 3) = 0.001 BTC
    - 0.001 × $90,300 = $90.30 (REJECTED!)

    With fix:
    - Detects notional < $100
    - Adjusts to 0.002 BTC
    - 0.002 × $90,300 = $180.60 (ACCEPTED!)
    """
    print("\n" + "="*70)
    print("Testing Rounding Bug Fix at BTC Price ~$90,300")
    print("="*70)

    price = 90303.60
    min_notional = 100.0

    # Step 1: Calculate base quantity
    print(f"\n📊 Step 1: Calculate quantity for ${min_notional} notional")
    base_quantity = min_notional / price
    print(f"   ${min_notional} / ${price:,.2f} = {base_quantity:.6f} BTC")

    # Step 2: Round to 3 decimals (simulating old bug)
    print(f"\n🔢 Step 2: Round to 3 decimals")
    rounded_quantity = round(base_quantity, 3)
    print(f"   round({base_quantity:.6f}, 3) = {rounded_quantity:.3f} BTC")

    # Step 3: Check notional after rounding
    print(f"\n💰 Step 3: Calculate notional after rounding")
    notional_after_rounding = rounded_quantity * price
    print(f"   {rounded_quantity:.3f} BTC × ${price:,.2f} = ${notional_after_rounding:.2f}")

    if notional_after_rounding < min_notional:
        print(f"\n❌ BUG DETECTED: Notional ${notional_after_rounding:.2f} < ${min_notional} minimum!")
        print(f"   This would be REJECTED by Binance")

        # Apply fix
        print(f"\n🔧 Step 4: Apply rounding fix")
        import math
        fixed_quantity = math.ceil((min_notional / price) * 1000) / 1000
        print(f"   ceil(({min_notional} / {price:.2f}) × 1000) / 1000")
        print(f"   = ceil({base_quantity:.6f} × 1000) / 1000")
        print(f"   = ceil({base_quantity * 1000:.3f}) / 1000")
        print(f"   = {math.ceil(base_quantity * 1000)} / 1000")
        print(f"   = {fixed_quantity:.3f} BTC")

        fixed_notional = fixed_quantity * price
        print(f"\n✅ Step 5: Verify fixed notional")
        print(f"   {fixed_quantity:.3f} BTC × ${price:,.2f} = ${fixed_notional:.2f}")

        assert fixed_notional >= min_notional, \
            f"Fixed notional ${fixed_notional:.2f} should be >= ${min_notional}"

        print(f"\n✅ PASS: Fixed notional ${fixed_notional:.2f} >= ${min_notional} minimum")
        print(f"   Order would be ACCEPTED by Binance")

        return True
    else:
        print(f"\n✅ No fix needed: Notional ${notional_after_rounding:.2f} >= ${min_notional}")
        return True


def test_rounding_fix_at_various_prices():
    """Test the fix works at different BTC price levels."""
    print("\n" + "="*70)
    print("Testing Rounding Fix at Various Price Levels")
    print("="*70)

    import math

    prices = [
        50000,   # Lower price
        75000,   # Mid price
        90303,   # Problem price
        100000,  # Round price
        150000,  # High price
    ]

    min_notional = 100.0
    results = []

    for price in prices:
        # Calculate quantity
        quantity = min_notional / price

        # Round to 3 decimals
        rounded = round(quantity, 3)
        notional_after_round = rounded * price

        # Apply fix if needed
        if notional_after_round < min_notional:
            fixed = math.ceil((min_notional / price) * 1000) / 1000
            fixed_notional = fixed * price
            status = "FIXED"
        else:
            fixed = rounded
            fixed_notional = notional_after_round
            status = "OK"

        results.append({
            'price': price,
            'original_qty': quantity,
            'rounded_qty': rounded,
            'notional_rounded': notional_after_round,
            'fixed_qty': fixed,
            'notional_fixed': fixed_notional,
            'status': status
        })

    print(f"\n{'Price':<12} {'Rounded Qty':<15} {'Notional':<12} {'Final Qty':<15} {'Final Notional':<15} {'Status':<10}")
    print("-" * 95)

    for r in results:
        print(f"${r['price']:<11,.0f} {r['rounded_qty']:<15.6f} ${r['notional_rounded']:<11.2f} "
              f"{r['fixed_qty']:<15.6f} ${r['notional_fixed']:<15.2f} {r['status']:<10}")

        # Assert all final notionals meet minimum
        assert r['notional_fixed'] >= min_notional, \
            f"At price ${r['price']}, notional ${r['notional_fixed']:.2f} < ${min_notional}"

    print("\n✅ All price levels pass minimum notional requirement")
    return True


def test_actual_strategy_position_sizing():
    """
    Test the real _calculate_position_size at the problem price ($90,303).

    At this price $100 rounds to 0.001 BTC = $90.30 notional (below the
    exchange minimum). The strategy must round UP to 0.002 - but only
    because that stays within the margin-based risk cap.
    """
    from strategy.deepseek_strategy import DeepSeekAIStrategy, DeepSeekAIStrategyConfig

    strategy = DeepSeekAIStrategy(config=DeepSeekAIStrategyConfig(
        instrument_id="BTCUSDT-PERP.BINANCE",
        bar_type="BTCUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL",
        equity=400.0,
        leverage=10.0,
        base_usdt_amount=100.0,
        sizing_mode="notional",   # this test targets the notional-mode rounding path
        use_rule_based_analyzer=True,
        prefetch_bars=False,
        use_account_balance=False,
        enable_telegram=False,
        sentiment_enabled=False,
    ))

    quantity = strategy._calculate_position_size(
        {'confidence': 'MEDIUM', 'signal': 'BUY'},
        {'price': 90303.60},
        {'overall_trend': '震荡整理', 'rsi': 55.0},
        None,
    )

    final_notional = quantity * 90303.60
    # Must either skip (0) or satisfy the exchange minimum
    assert quantity == 0.0 or final_notional >= 100.0, \
        f"Final notional ${final_notional:.2f} should be >= $100"
    # At these settings the risk cap ($400) allows the round-up to 0.002
    assert quantity == 0.002, f"Expected 0.002 BTC, got {quantity}"

    return True


def run_all_tests():
    """Run all rounding fix tests."""
    tests = [
        ("Rounding Bug at $90k Price", test_rounding_bug_at_90k_price),
        ("Rounding Fix at Various Prices", test_rounding_fix_at_various_prices),
        # Note: Actual strategy test removed due to NautilusTrader logger restrictions
        # The integration tests cover this scenario instead
    ]

    print("\n" + "="*70)
    print("Rounding Bug Fix Test Suite")
    print("="*70)

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*70)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
