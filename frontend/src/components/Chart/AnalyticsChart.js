import React, { useState, useEffect } from 'react'
import Button from '../Button/Button';
import { linechart } from '../../utils/Icons';
import styled from 'styled-components'
import Slider, { Range } from 'rc-slider';
import 'rc-slider/assets/index.css';
import {Chart as ChartJs, 
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
} from 'chart.js'

import { InnerLayout } from '../../styles/layouts'
import {Line} from 'react-chartjs-2'
import { useGlobalContext } from '../../context/globalContext'
import { dateFormat } from '../../utils/dateFormat'
import 'chartjs-plugin-datalabels';

ChartJs.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    ArcElement,
)

/*
<div className="selects input-control">
                            <select required value={model} name="category" id="category" onChange={handleInput('model')}>
                                <option value=""  disabled >Select Option</option>
                                <option value="ANN">ANN</option>
                                <option value="GAN">GAN</option>
                                <option value="LSTM">LSTM</option> 
                            </select>
                        </div>
<div className="input-control">
                            <input 
                                type="text"
                                value={lback}
                                name={'lback'}
                                placeholder='Look Back'
                                onChange={handleInput('lookback')}
                            />
                        </div>
*/


function AnalyticsChart() {

    const {preds, getPrediction, lastm, incLen, expLen} = useGlobalContext()

    const [selectedValue, setSelectedValue] = useState("expense");

    const [inputState, setInputState] = useState({
        choice: 'expense',
        model: 'LSTM',
        months: '12',
        lookback: '12',
        ifTrain: 'True',
    })

    function generateDataArray(incomeMonth, expenseMonth, incomeArray, expenseArray, X) {
        // Convert input months to Date objects
        const incomeDate = parseMonthYear(incomeMonth);
        const expenseDate = parseMonthYear(expenseMonth);
    
        // Determine the start and end months for the new array
        const startDate = new Date(Math.min(incomeDate, expenseDate));
        startDate.setMonth(startDate.getMonth() + 1);
        const endDate = new Date(Math.max(incomeDate, expenseDate));
        endDate.setMonth(endDate.getMonth() + X);
    
        // Generate the date range
        const dataArray = [];
        let currentDate = new Date(startDate);
        while (currentDate <= endDate) {
            const month = currentDate.getMonth() + 1; // Months are 0-indexed
            const year = currentDate.getFullYear();
            dataArray.push(`${month}/${year}`);
            currentDate.setMonth(currentDate.getMonth() + 1);
        }
    
        // Pad income and expense arrays
        const paddedIncomeArray = padArray(incomeArray, dataArray.length - incomeArray.length, (incomeDate < expenseDate));
        const paddedExpenseArray = padArray(expenseArray, dataArray.length - expenseArray.length, (incomeDate > expenseDate));
    
        // Log results
        console.log('Data Array:', dataArray);
        console.log('Padded Income Array:', paddedIncomeArray);
        console.log('Padded Expense Array:', paddedExpenseArray);
    
        return {
            dataArray,
            paddedIncomeArray,
            paddedExpenseArray
        };
    }
    
    function parseMonthYear(monthYear) {
        const [month, year] = monthYear.split('/');
        return new Date(`${year}-${month}-01`);
    }
    
    function padArray(arr, paddingCount, appendToEnd) {
        // Clone the original array
        const paddedArray = arr.slice();
    
        // Pad with zeroes
        for (let i = 0; i < paddingCount; i++) {
            if (appendToEnd) {
                paddedArray.push(0);
            } else {
                paddedArray.unshift(0);
            }
        }
    
        return paddedArray;
    }

    function getNextXMonthsFromDate(startDate, numMonths) {
        const result = [];
        const [startMonth, startYear] = startDate.split('/').map(Number);
        let date = new Date(startYear, startMonth - 1); // Months are 0-indexed in JavaScript dates
    
        for (let i = 0; i < numMonths; i++) {
            date.setMonth(date.getMonth() + 1);
            const month = date.getMonth() + 1;
            const year = date.getFullYear();
            const formattedDate = `${month}/${year}`;
            result.push(formattedDate);
        }
    
        return result;
    }

    function generateBothChartData() {
        const {
          dataArray,
          paddedIncomeArray,
          paddedExpenseArray
        } = generateDataArray(lastm[1], lastm[0], preds[1], preds[0], parseInt(inputState.months));
      
        // Filter non-zero values along with their x-axis labels
        const filteredIncomeArray = paddedIncomeArray.reduce((acc, value, index) => {
          if (value !== 0) {
            acc.push({
              x: dataArray[index],
              y: value
            });
          }
          return acc;
        }, []);
      
        const filteredExpenseArray = paddedExpenseArray.reduce((acc, value, index) => {
          if (value !== 0) {
            acc.push({
              x: dataArray[index],
              y: value
            });
          }
          return acc;
        }, []);
      
        const data = {
          labels: dataArray,
          datasets: [
            {
              label: "Expense",
              data: filteredExpenseArray,
              backgroundColor: '#FD0000',
              borderColor: '#FD0000',
              tension: 0.2,
              showLine: true, // Hide the line connecting points
              pointRadius: 4, // Increase the point size for better visibility
            },
            {
              label: "Income",
              data: filteredIncomeArray,
              backgroundColor: '#00a23c',
              borderColor: '#00a23c',
              tension: 0.2,
              showLine: true, // Hide the line connecting points
              pointRadius: 4, // Increase the point size for better visibility
            }
          ]
        };
      
        const options = {
          scales: {
            x: {
              title: {
                display: true,
                text: 'Next ' + inputState.months + " Months"
              }
            },
            y: {
              title: {
                display: true,
                text: "Prediction for both"
              }
            }
          },
          plugins: {
            title: {
              display: true,
              text: "Predicted Incomes and expenses for the next " + inputState.months + " months"
            }
          }
        };
      
        return {"data" : data, "options": options};
      }
      
    

    const ex_cat = [
        "Education",
        "Groceries",
        "Health",
        "Subscriptions",
        "Takeaways",
        "Clothing",
        "Travelling",
        "Other",
    ]

    const in_cat = [
        "Salary",
        "Freelancing",
        "Investments",
        "Stocks",
        "Bitcoin",
        "Bank Transfer",
        "Youtube",
        "Other",
    ]

    const cat = inputState.choice === "income"?in_cat:ex_cat
    const ch = inputState.choice === "income"?"Income":"Expense"

    function generateChartData() {
        //const datagen = generateData(inputState.choice, inputState.choice === "income"?incomes:expenses)
        const amount = preds
        const mths = getNextXMonthsFromDate(lastm, parseInt(inputState.months, 10))
        const data = {
            labels: mths,
            datasets: [
                {
                    label: cat[0],
                    data: amount[0],
                    backgroundColor: '#FF5733',
                    borderColor: '#FF5733',
                    tension: .2
                },

                {
                    label: cat[1],
                    data: amount[1],
                    backgroundColor: '#33FF57',
                    borderColor: '#33FF57',
                    tension: .2
                },

                {
                    label: cat[2],
                    data: amount[2],
                    backgroundColor: '#5733FF',
                    borderColor: '#5733FF',
                    tension: .2
                },

                {
                    label: cat[3],
                    data: amount[3],
                    backgroundColor: '#FF33A1',
                    borderColor: '#FF33A1',
                    tension: .2
                },

                {
                    label: cat[4],
                    data: amount[4],
                    backgroundColor: '#33A1FF',
                    borderColor: '#33A1FF',
                    tension: .2
                },

                {
                    label: cat[5],
                    data: amount[5],
                    backgroundColor: '#A1FF33',
                    borderColor: '#A1FF33',
                    tension: .2
                },

                {
                    label: cat[6],
                    data: amount[6],
                    backgroundColor: '#FFD700',
                    borderColor: '#FFD700',
                    tension: .2
                },

                {
                    label: cat[7],
                    data: amount[7],
                    backgroundColor: '#8A2BE2',
                    borderColor: '#8A2BE2',
                    tension: .2
                }

            ]
        }

        const options = {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Next ' + amount[0].length + " Months"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: ch + " Prediction"
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: "Predicted " + ch + " for the next " + amount[0].length + " months"
                }
            }
        };


        return {"data" : data, "options": options}
    }

    let data = null;
    if (preds != -1) {
        if (inputState.choice !== "both"){
            data = generateChartData(); // Replace with your actual function to generate chart data
        } else {
            
            data = generateBothChartData();
        }
    }

    

    const {model, months, lback, choice} = inputState;
    //const [model, setModel] = useState("LSTM");
    //const [months, setMonths] = useState(0);
    //const [lback, setLback] = useState(0);
  
    const handleRadioChange = ( 
        value 
    ) => { 
        setSelectedValue(value);
        setInputState({...inputState, "choice": value})
    }; 

    const handleInput = name => e => {
        setInputState({...inputState, [name]: e.target.value})
    }

    const handleSubmit = e => {
        e.preventDefault()
        console.log(inputState);
        getPrediction(inputState)
        console.log(preds); 
    }

    const OnChangeEventTriggerd = (value) => {
        setInputState({...inputState, "months": value})
    };

    return (
        <InnerLayout>
            <AnalyticsLayout>
                {data ? (
                    <AnalyticsChartStyled>
                        <Line data={data["data"]} options={data["options"]} />
                    </AnalyticsChartStyled>
                    ) : (
                    <AnalyticsChartStyled style={{ 
                        background: '#FFFFFF', 
                        display: 'flex', 
                        justifyContent: 'center', 
                        alignItems: 'center', 
                        color: '#000000', 
                        height: 'auto',
                        textAlign: 'center',
                        padding: "0px 100px"
                        }}>
                        Generate the predictions chart or check if the number of months is at max 12
                    </AnalyticsChartStyled>
                )}
                <OptionStyled>
                    <div style={Radiostyles.radioGroup}>
                        <div style={Radiostyles.radioButton}>
                            <input
                            type="radio"
                            id="income"
                            value="income"
                            checked={selectedValue === "income"}
                            onChange={() => handleRadioChange("income")}
                            />
                            <label htmlFor="income" style={Radiostyles.radioLabel}>
                            Income
                            </label>
                        </div>

                        <div style={Radiostyles.radioButton}>
                            <input
                            type="radio"
                            id="expense"
                            value="expense"
                            checked={selectedValue === "expense"}
                            onChange={() => handleRadioChange("expense")}
                            />
                            <label htmlFor="expense" style={Radiostyles.radioLabel}>
                            Expense
                            </label>
                        </div>

                        <div style={Radiostyles.radioButton}>
                            <input
                            type="radio"
                            id="both"
                            value="both"
                            checked={selectedValue === "both"}
                            onChange={() => handleRadioChange("both")}
                            />
                            <label htmlFor="both" style={Radiostyles.radioLabel}>
                            Both
                            </label>
                        </div>
                    </div>
                    <FormStyled>
                        
                        <div className="input-control">
                            <input 
                                type="text"
                                value={months}
                                name={'months'}
                                placeholder='Number of months'
                                onChange={handleInput('months')}
                            />
                        </div>
                        <div style={containerStyle}>
                            <Slider
                                value={inputState.months}
                                step={1}
                                min={0}
                                max={12}
                                onChange={OnChangeEventTriggerd}
                                handleStyle={handleStyle}
                                railStyle={railStyle}
                                trackStyle={trackStyle}
                            />
                        </div>
                        
                        <div className="submit-btn">
                            <Button 
                                name={'Generate Chart'}
                                icon={linechart}
                                bPad={'.8rem 1.6rem'}
                                bRad={'30px'}
                                bg={'var(--color-accent'}
                                color={'#fff'}
                                onClick={handleSubmit}
                            />
                        </div>
                    </FormStyled>
                </OptionStyled>
            </AnalyticsLayout>
        </InnerLayout>
    )
}

const containerStyle = {
    width: '90%', // Set the desired width for the slider container
    margin: '0 auto', // Center the container horizontally (optional)
  };

const handleStyle = {
    backgroundColor: '#3eb499', // Change the color of the handle
    borderColor: '#3eb499', // Change the border color of the handle
    opacity: 1,
  };
  
  const railStyle = {
    backgroundColor: '#f56692', // Change the color of the rail
  };
  
  const trackStyle = {
    backgroundColor: '#3eb499', // Change the color of the track
  };

const Radiostyles = { 
    container: { 
        display: "flex",
        flex: 0, 
        justifyContent: "center", 
        alignItems: "center", 
    }, 
    heading: { 
        color: "green", 
        textAlign: "center", 
    }, 
    radioGroup: {
        display: "flex", 
        flexDirection: "row", 
        alignItems: "center", 
        justifyContent: 
            "space-around", 
        margin: "0px 0px 20px 20px", 
        borderRadius: "8px", 
        backgroundColor: "white", 
        padding: "10px",
    }, 
    radioButton: { 
        display: "flex", 
        flexDirection: "row", 
        alignItems: "center", 
    }, 
    radioLabel: { 
        marginLeft: "10px",
        marginRight: "10px",
        fontSize: "20px", 
        color: "#222260",
    }, 
};

const AnalyticsLayout = styled.div`
    display: flex;
    flexDirection: row;
`

const OptionStyled = styled.div`
    width: "30%";
`

const AnalyticsChartStyled = styled.div`
    background: #FCF6F9;
    border: 2px solid #FFFFFF;
    box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
    border-radius: 20px;
    height: 100%;
    width: 70%;
`;

const FormStyled = styled.form`
    display: flex;
    flex-direction: column;
    gap: 1.9rem;
    padding: 0.7rem 0rem .5rem 1.4rem;
    input, textarea, select{
        font-family: inherit;
        font-size: inherit;
        width: 100%;
        outline: none;
        border: none;
        padding: .5rem 1rem;
        border-radius: 5px;
        border: 2px solid #fff;
        background: transparent;
        resize: none;
        box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
        color: rgba(34, 34, 96, 0.9);
        &::placeholder{
            color: rgba(34, 34, 96, 0.4);
        }
    }


    .selects{
        display: flex;
        select{
            color: rgba(34, 34, 96, 0.4);
            &:focus, &:active{
                color: rgba(34, 34, 96, 1);
            }
        }
    }

    .submit-btn{
        button{
            width: 100%;
            box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
            &:hover{
                background: var(--color-green) !important;
            }
        }
    }
`;

export default AnalyticsChart