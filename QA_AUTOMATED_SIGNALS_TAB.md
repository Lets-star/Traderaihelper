# QA Checklist: Automated Signals Tab

## Overview
This document provides a comprehensive QA checklist for testing the new "Automated Signals" (🤖) tab in the web UI.

## Prerequisites
- Running Streamlit web UI: `streamlit run web_ui.py`
- Sample data with trading system signals (JSON format)
- Understanding of trading system signal structure

## UI Display Tests

### Tab Navigation
- [ ] Tab "🤖 Automated Signals" appears in the tab bar
- [ ] Tab is positioned after "Trade Signals" and before "Astrology"
- [ ] Tab can be clicked and is accessible
- [ ] Tab maintains state when switching between other tabs

### No Data Scenarios
- [ ] When no signals data available: Warning message displays
- [ ] When advanced.trade_plan exists but no detailed signals: Info message displays
- [ ] Help text suggests running trading system analyzer
- [ ] All messaging is user-friendly and informative

## Signal Display Tests

### Signal Type Display
- [ ] BUY signal displays with 🟢 emoji and green styling
- [ ] SELL signal displays with 🔴 emoji and red styling
- [ ] NEUTRAL signal displays with ⚪ emoji and neutral styling
- [ ] Signal type text is clearly visible and readable

### Confidence Metric
- [ ] Confidence displays as percentage (0-100%)
- [ ] Confidence >= 70%: Shows "🟢 High" indicator
- [ ] Confidence 50-70%: Shows "🟡 Medium" indicator
- [ ] Confidence < 50%: Shows "⚪ Low" indicator
- [ ] Confidence value is accurate to 1 decimal place

### Timestamp Display
- [ ] Signal timestamp displays in HH:MM:SS format
- [ ] Timestamp is readable and correctly formatted
- [ ] Missing timestamps handled gracefully

## Factor Analysis Tests

### Factor Table Display
- [ ] Factors table displays when factors are present
- [ ] Factors table is hidden when no factors
- [ ] Table columns: Factor, Score, Weight, Emoji, Description
- [ ] Factor names are displayed correctly
- [ ] Factor scores displayed with 2 decimal places
- [ ] Factor weights displayed with 2 decimal places
- [ ] Emoji column shows emoji or ⚪ if missing
- [ ] Descriptions are complete and readable

### Factor Data Validation
- [ ] Factor scores in reasonable range (0-100)
- [ ] Factor weights are positive
- [ ] At least one factor typically present for non-NEUTRAL signals
- [ ] Factor descriptions are meaningful when present

## Position Plan Tests

### Position Metrics Display
- [ ] Entry price displays in proper format ($X.XXXX)
- [ ] Position size displays in USD ($X.XX)
- [ ] Direction displays as LONG/SHORT/FLAT
- [ ] Leverage displays with 'x' suffix (e.g., 10.0x)
- [ ] All four metrics visible in columns

### TP/SL Ladder Display
- [ ] Stop Loss displays with proper price format ($X.XXXX)
- [ ] Risk distance displays in USD and percentage
- [ ] Take Profit levels numbered (TP1, TP2, TP3, etc.)
- [ ] Each TP level shows price and percentage profit
- [ ] Profit percentages show + sign and 2 decimals
- [ ] Message shown when no TP levels defined
- [ ] Stop Loss always less than Entry for LONG positions
- [ ] Stop Loss always greater than Entry for SHORT positions

### Risk Metrics Display
- [ ] Risk/Reward Ratio displays when present
- [ ] Max Risk % displays as percentage when present
- [ ] Metrics formatted clearly and readable
- [ ] Metrics hidden when not defined

### Position Plan Validation
- [ ] Entry price is positive
- [ ] Stop loss is positive
- [ ] Take profit levels are all positive
- [ ] Position size is positive
- [ ] Leverage is positive
- [ ] Position direction is valid (long/short/flat)

## Holding Horizon Tests

- [ ] Holding horizon section displays when present
- [ ] Holding horizon value is positive integer
- [ ] Displays in info box with clear label
- [ ] Section hidden when not present
- [ ] Format: "**Estimated Holding Period:** X bars"

## Signal Rationale Tests

### Rationale Display
- [ ] Signal Rationale section displays when explanation present
- [ ] Primary reason displays clearly
- [ ] Supporting factors display as bullet list
- [ ] Risk factors display as bullet list with ⚠️ emoji
- [ ] Market context displays when present
- [ ] All text is readable and not truncated

### Rationale Content Validation
- [ ] Primary reason is non-empty string
- [ ] Supporting factors (if any) are strings
- [ ] Risk factors (if any) are strings
- [ ] Market context includes relevant information

## Cancellation Reasons Tests

- [ ] Section displays only when cancellation_reasons present
- [ ] Section uses warning styling (yellow background)
- [ ] Reasons display as bullet list
- [ ] Each reason is clearly readable
- [ ] Multiple reasons display in order
- [ ] Clear visual distinction from other sections

## Performance Metrics Tests

### Metrics Display
- [ ] Performance Metrics section displays when optimization_stats present
- [ ] Win Rate displays in percentage format (0-100%)
- [ ] Profit Factor displays with 2 decimal places
- [ ] Sharpe Ratio displays with 2 decimal places
- [ ] Total Signals displays as positive integer
- [ ] Avg Profit % displays with 2 decimals and % symbol
- [ ] Avg Loss % displays with 2 decimals and % symbol
- [ ] Profitable/Losing signal count displays as "X/Y"

### Performance Metrics Validation
- [ ] Win Rate between 0-100
- [ ] Profit Factor >= 0
- [ ] Sharpe Ratio is numeric
- [ ] Total Signals > 0
- [ ] Profitable + Losing = Total (when both present)
- [ ] Profitable >= 0 and Losing >= 0
- [ ] Avg Profit % usually positive for winning signals
- [ ] Avg Loss % usually negative for losing signals

## Data Format Tests

### JSON Payload Structure
- [ ] Payload can contain "automated_signals" key
- [ ] Payload can contain "trading_signals" key
- [ ] Either key location is recognized
- [ ] Payload without signals is handled gracefully

### Signal Object Structure
- [ ] signal_type field present and valid
- [ ] confidence field present (0-1 float)
- [ ] timestamp field present (milliseconds)
- [ ] symbol field present
- [ ] timeframe field present
- [ ] factors array present (can be empty)
- [ ] position_plan object present or null
- [ ] explanation object present or null
- [ ] optimization_stats object present or null

### Position Plan Object Structure
- [ ] entry_price field present
- [ ] stop_loss field present (can be null)
- [ ] take_profit_levels array present (can be empty)
- [ ] position_size_usd field present (can be null)
- [ ] risk_reward_ratio field present (can be null)
- [ ] max_risk_pct field present (can be null)
- [ ] leverage field present (can be null)
- [ ] direction field present (can be null)

### Explanation Object Structure
- [ ] primary_reason field present
- [ ] supporting_factors array present
- [ ] risk_factors array present
- [ ] market_context field present (can be null)

### Optimization Stats Object Structure
- [ ] backtest_win_rate field present (can be null)
- [ ] avg_profit_pct field present (can be null)
- [ ] avg_loss_pct field present (can be null)
- [ ] sharpe_ratio field present (can be null)
- [ ] total_signals field present
- [ ] profitable_signals field present
- [ ] losing_signals field present

## Edge Cases

### Missing Data Handling
- [ ] Null position_plan handled gracefully
- [ ] Null explanation handled gracefully
- [ ] Null optimization_stats handled gracefully
- [ ] Missing factors handled gracefully
- [ ] Empty factors array handled gracefully
- [ ] Missing TP levels handled gracefully
- [ ] Missing description in factors handled gracefully

### Extreme Values
- [ ] Very high confidence (0.99) displays correctly
- [ ] Very low confidence (0.01) displays correctly
- [ ] Very large prices ($1,000,000) format correctly
- [ ] Very small prices ($0.0001) format correctly
- [ ] Very large position sizes ($100,000) format correctly
- [ ] Very high leverage (100x) displays correctly
- [ ] Extreme win rates (100% / 0%) handled correctly
- [ ] Extreme profit factors (10+) display correctly

### Unicode and Special Characters
- [ ] Emoji display correctly (🟢 🔴 ⚪ ⚠️ 📊 💼 ⏱️ 📝 ⛔ 📈 💡)
- [ ] Special characters in descriptions display correctly
- [ ] Long descriptions don't break layout
- [ ] Non-ASCII characters handled gracefully

## Visual Layout Tests

### Responsive Design
- [ ] Tab content displays properly on wide screens (1920px+)
- [ ] Tab content displays properly on standard screens (1280px)
- [ ] Tab content displays properly on tablets (800px)
- [ ] No horizontal scrolling needed
- [ ] Columns properly stack/resize

### Spacing and Alignment
- [ ] Clear separation between sections (horizontal lines)
- [ ] Consistent spacing between elements
- [ ] Metric boxes properly aligned
- [ ] Tables have good readability
- [ ] Text is not cramped

### Color and Styling
- [ ] Green styling for positive/BUY signals
- [ ] Red styling for negative/SELL signals
- [ ] Blue/info styling for neutral states
- [ ] Warning boxes properly highlighted
- [ ] All text is readable with sufficient contrast

## Integration Tests

### Tab Integration
- [ ] Tab coexists with other tabs without issues
- [ ] Switching between tabs is smooth
- [ ] No data bleeding between tabs
- [ ] Session state preserved when tab hidden/shown

### Data Flow
- [ ] Data from collector payload properly passed to tab
- [ ] Web UI can read JSON signal format
- [ ] No errors when parsing signal JSON
- [ ] Graceful handling of malformed JSON

### Export Functionality
- [ ] Exported JSON includes full signal data
- [ ] Exported JSON can be re-imported
- [ ] JSON export contains all necessary fields

## Performance Tests

- [ ] Tab loads within 2 seconds with data
- [ ] Tab renders smoothly without lag
- [ ] Large factor lists (10+) render without performance issues
- [ ] Multiple signals data processed efficiently

## Accessibility Tests

- [ ] Tab title is descriptive ("🤖 Automated Signals")
- [ ] Metric labels are clear and descriptive
- [ ] Color not solely used to convey information (emoji/text also used)
- [ ] Text has sufficient size for readability
- [ ] Buttons/interactive elements have clear labels

## Documentation Tests

- [ ] Help text is clear and actionable
- [ ] Disclaimer about trading system present
- [ ] Instructions for enabling full functionality provided
- [ ] No typos or grammatical errors in text

## Test Scenarios

### Scenario 1: Complete BUY Signal
**Given:** Full BUY signal with all fields populated
**When:** Tab is viewed
**Then:**
- [ ] Green BUY indicator displays
- [ ] Confidence >= 70% shows "🟢 High"
- [ ] All factors display with descriptions
- [ ] Complete position plan with TP levels visible
- [ ] Rationale shows supporting and risk factors
- [ ] Performance metrics display

### Scenario 2: SELL Signal with Rejection
**Given:** SELL signal that was rejected
**When:** Tab is viewed
**Then:**
- [ ] Red SELL indicator displays
- [ ] Cancellation reasons section visible
- [ ] No position plan displayed
- [ ] Risk factors clearly shown
- [ ] User understands why signal was rejected

### Scenario 3: NEUTRAL with No Position
**Given:** NEUTRAL signal with no position plan
**When:** Tab is viewed
**Then:**
- [ ] Neutral indicator (⚪) displays
- [ ] Position plan section not visible
- [ ] Rationale explains waiting for setup
- [ ] Tab doesn't show error
- [ ] User understands to check back later

### Scenario 4: High Confidence vs Low Confidence
**Given:** Two signals with different confidence levels
**When:** Tab displays both signals
**Then:**
- [ ] High confidence (0.9) shows "🟢 High" with 90.0%
- [ ] Low confidence (0.3) shows "⚪ Low" with 30.0%
- [ ] Visual difference is clear

### Scenario 5: No Signals Available
**Given:** Payload without any trading signals
**When:** Tab is viewed
**Then:**
- [ ] Warning message displays
- [ ] No errors in console
- [ ] User knows what to do next
- [ ] Tab gracefully degrades

## Manual Testing Instructions

1. **Set up test environment:**
   ```bash
   cd /home/engine/project
   streamlit run web_ui.py
   ```

2. **Load test data with signals:**
   - Navigate to the web UI
   - Select a token and timeframe
   - Ensure payload includes `automated_signals` or `trading_signals`

3. **Test each section:**
   - Verify signal display
   - Check factor analysis
   - Validate position plan
   - Review performance metrics

4. **Test edge cases:**
   - Try with missing fields
   - Try with extreme values
   - Try with no signals data

5. **Verify visual layout:**
   - Check on different screen sizes
   - Verify all elements visible
   - Check text readability

## Automated Test Coverage

The project includes comprehensive unit tests in `tests/test_web_ui_automated_signals.py`:

**Test Classes:**
1. `TestTradingSignalPayloadStructure` - JSON serialization/deserialization
2. `TestWebUIPayloadFormats` - Various payload formats
3. `TestSignalMetricsCalculations` - Metric calculations
4. `TestWebUIDisplayFormatting` - Display formatting
5. `TestWebUIDataValidation` - Data validation
6. `TestSignalFactorAnalysis` - Factor analysis
7. `TestSignalExplanationDisplay` - Explanation display
8. `TestPerformanceMetricsDisplay` - Performance metrics

**Running Tests:**
```bash
cd /home/engine/project
python -m pytest tests/test_web_ui_automated_signals.py -v
```

**Test Coverage:**
- Signal serialization/deserialization: ✓
- Payload format variations: ✓
- Metrics calculations: ✓
- Display formatting: ✓
- Data validation: ✓
- Edge case handling: ✓

## Regression Tests

After implementation, verify:
- [ ] No errors in other tabs
- [ ] Data flow not affected
- [ ] Performance unchanged
- [ ] Export functionality still works
- [ ] Multi-timeframe analysis unaffected
- [ ] All existing tabs functional

## Known Limitations / Future Improvements

1. **Current State:** Tab gracefully handles missing trading system signals
2. **Enhancement:** Live signal updates with WebSocket
3. **Enhancement:** Historical signal replay
4. **Enhancement:** Signal backtest results integration
5. **Enhancement:** Interactive position sizing calculator
6. **Enhancement:** Alert notifications for new signals

## Sign-off

| Role | Name | Date | Status |
|------|------|------|--------|
| Developer | [Your Name] | YYYY-MM-DD | ✓ Complete |
| QA Lead | [QA Lead] | YYYY-MM-DD | [ ] Approved |
| Product Owner | [PO] | YYYY-MM-DD | [ ] Approved |

## Notes

- The tab is fully functional with trading system JSON signals
- Graceful degradation when signals not available
- Compatible with existing web UI infrastructure
- No breaking changes to other tabs
- Clear user guidance for signal interpretation

---

**Last Updated:** 2024
**Version:** 1.0
