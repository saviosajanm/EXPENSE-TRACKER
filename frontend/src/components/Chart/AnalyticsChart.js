import React, { useState, useEffect } from 'react'
import Button from '../Button/Button';
import { linechart } from '../../utils/Icons';
import styled from 'styled-components'
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

function AnalyticsChart() {

    const {preds, getPrediction, lastm} = useGlobalContext()

    const [selectedValue, setSelectedValue] = useState("expense");
    const [inputState, setInputState] = useState({
        choice: 'expense',
        model: 'LSTM',
        months: '',
        lookback: '',
    })

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

    function generateData(choice, data) {
        // Initialize arrays A and B
        const A = [];
        const B = [];

        // Create a map to store monthly expenses
        const monthlydataMap = {};

        // Iterate through expenses and populate monthlyExpensesMap
        data.forEach(i => {
            const date = new Date(i.date);
            const monthYearKey = `${date.getMonth() + 1}/${date.getFullYear()}`;

            if (!monthlydataMap[monthYearKey]) {
                monthlydataMap[monthYearKey] = Array(8).fill(0);
            }

            if (choice === "income") {
                var categoryIndex = [
                    "salary", "freelancing", "investments", "stocks", "bitcoin", "bank", "youtube", "other",
                ].indexOf(i.category.toLowerCase());
            } else
            if (choice === "expense") {
                var categoryIndex = [
                    'education', 'groceries', 'health', 'subscriptions', 'takeaways', 'clothing', 'travelling', 'other'
                ].indexOf(i.category.toLowerCase());
            }


            if (categoryIndex !== -1) {
                monthlydataMap[monthYearKey][categoryIndex] += i.amount;
            }
        });

        // Get the start and end months
        const startDate = new Date(data[0].date);
        const endDate = new Date(data[data.length - 1].date);

        // Populate arrays A and B for continuous months
        for (let currentMonth = new Date(startDate); currentMonth <= endDate; currentMonth.setMonth(currentMonth.getMonth() + 1)) {
        const monthYearKey = `${currentMonth.getMonth() + 1}/${currentMonth.getFullYear()}`;

        if (!monthlydataMap[monthYearKey]) {
            A.push(Array(8).fill(0));
            B.push(monthYearKey);
        } else {
            A.push(monthlydataMap[monthYearKey]);
            B.push(monthYearKey);
        }
        }
        let X = A[0].map((_, colIndex) => A.map(row => row[colIndex]));
        // Display the results
        console.log("Array A:", X);
        console.log("Array B:", B);
        return {"amount":X, "months":B}
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
        data = generateChartData(); // Replace with your actual function to generate chart data
    }
    //console.log(preds, "-----------++++++++++++++++++++");

    

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
                        Generate the predictions chart or check for faulty parameters that might not work with the chosen model and/or data
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
                                value={months}
                                name={'months'}
                                placeholder='Number of months'
                                onChange={handleInput('months')}
                            />
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