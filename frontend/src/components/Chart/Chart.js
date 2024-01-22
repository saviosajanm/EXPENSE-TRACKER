import React from 'react'
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

import {Line} from 'react-chartjs-2'
import { useGlobalContext } from '../../context/globalContext'
import { dateFormat } from '../../utils/dateFormat'

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

function Chart() {

    const {incomes, expenses} = useGlobalContext()

    let incomeList = [
        ...incomes.map((income) => {
            const {amount} = income
            return amount
        })
    ]

    let expenseList = [
        ...expenses.map((expense) => {
            const {amount} = expense
            return amount
        })
    ]

    const incomeLength = incomeList.length;
    const expenseLength = expenseList.length;

    if (incomeLength > expenseLength) {
        const diff = incomeLength - expenseLength;
        const zerosToAdd = Array(diff).fill(0);
        expenseList = expenseList.concat(zerosToAdd);
    } else {
        // Add zeros to the end of incomeList to make it equal in length to expenseList.
        const diff = expenseLength - incomeLength;
        const zerosToAdd = Array(diff).fill(0);
        incomeList = incomeList.concat(zerosToAdd);
    }

    const balanceList = incomeList.map(function(item, index) {
        return item - expenseList[index];
      })

    const data = {
        labels: incomes.map((inc) => {
            const {date} = inc
            return dateFormat(date)
        }),
        datasets: [
            {
                label: 'Income',
                data: incomeList,
                backgroundColor: 'green',
                tension: .2
            },

            {
                label: 'Expenses',
                data: expenseList,
                backgroundColor: 'red',
                tension: .2
            },

            {
                label: 'Balance',
                data: balanceList,
                backgroundColor: 'blue',
                tension: .2
            }

        ]
    }


    return (
        <ChartStyled>
            <Line data={data}/>
        </ChartStyled>
    )
}

const ChartStyled = styled.div`
    background: #FCF6F9;
    border: 2px solid #FFFFFF;
    box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
    padding: 1rem;
    border-radius: 20px;
    height: 100%;
`;

export default Chart