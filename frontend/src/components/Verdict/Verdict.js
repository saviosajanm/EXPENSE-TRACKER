import React, { useState, useEffect } from 'react'
import styled from 'styled-components'
import { InnerLayout } from '../../styles/layouts';
import DatePicker from 'react-datepicker'
import "react-datepicker/dist/react-datepicker.css";
import Button from '../Button/Button';
import { useGlobalContext } from '../../context/globalContext'
import { search } from '../../utils/Icons';
import { dollar } from '../../utils/Icons';

function Verdict() {

    const {preds, getPrediction, lastm, totalExpense, totalIncome, totalBalance} = useGlobalContext()

    const [inputState, setInputState] = useState({
        saving: '',
        date: '',
        model: 'LSTM',
    })

    const handleInput = name => e => {
        setInputState({...inputState, [name]: e.target.value})
    }

    const handleSubmit = e => {
        e.preventDefault()
    }

    const {saving, date, model} = inputState;

    return (
        <InnerLayout>
            <FormStyled>
                <div style={{display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%",}}>
                <div className="input-control">
                    <input 
                        type="text"
                        value={saving}
                        name={'amount'}
                        placeholder='Salary Amount'
                        onChange={handleInput('saving')}
                        autocomplete="off"
                    />
                    </div>
                    <div className="input-control">
                    <DatePicker 
                        id='date'
                        placeholderText="Enter A Date"
                        selected={date}
                        dateFormat="dd/MM/yyyy"
                        onChange={handleInput('date')}
                        autocomplete="off"
                    />
                    </div>
                    <div className="submit-btn">
                        <Button 
                            name={'Generate Verdict'}
                            icon={search}
                            bPad={'.8rem 1.6rem'}
                            bRad={'30px'}
                            bg={'#0e5093'}
                            color={'#fff'}
                            onClick={handleSubmit}
                        />
                    </div>
                </div>
                <div className="amount-con">
                    <div className="income">
                        <h2>Income</h2>
                        <p>
                        {dollar} {totalIncome()}
                        </p>
                    </div>
                    <div className="expense">
                        <h2>Expense</h2>
                        <p>
                        {dollar} {totalExpense()}
                        </p>
                    </div>
                    <div className="balance">
                        <h2>Balance</h2>
                        <p>
                        {dollar} {totalBalance()}
                        </p>
                    </div>
                </div>
                <div className="amount-verd">
                    <div className="balance">
                        <h2>Verdict:</h2>
                        <p>
                        This is a fake verdict. Do something about it.
                        </p>
                    </div>
                </div>
            </FormStyled>
        </InnerLayout>
    )
  }
  

  const FormStyled = styled.form`
    padding: 2rem .5rem .5rem .5rem;
    input, textarea, select{
        font-family: inherit;
        font-size: inherit;
        width: 300px;
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
            width: 300px;
            box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
            &:hover{
                background: #3eb499 !important;
            }
        }
    }

    .amount-con{
        width: 100%;
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        align-items: center;
        padding: 0rem 5rem;
        .income, .expense, .balance{
            flex: 1;
            background: #FCF6F9;
            border: 2px solid #FFFFFF;
            box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
            justify-content: center;
            align-items: center;
            border-radius: 50px;
            text-align: center;
            p{
                color: var(--color-green);
                font-size: 1.5rem;
                font-weight: 700;
            }
        }

        h2{
            display: inline;
            font-size: 1.3rem;
        }

        .balance{
            p{
                color: blue;
            }
        }
        .expense{
            p{
                color: red;
            }
        }
    }

    .amount-verd{
        width: 100%;
        display: flex;
        gap: 1.5rem;
        margin-top: 1rem;
        align-items: center;
        padding: 0rem 11rem;
        .balance{
            flex: 1;
            justify-content: center;
            align-items: center;
            border-radius: 50px;
            text-align: center;
            p{
                color: var(--color-green);
                font-size: 1.5rem;
                font-weight: 700;
            }
        }
        h2{
            display: inline;
            font-size: 1.3rem;
        }
    }
`;
  
  export default Verdict