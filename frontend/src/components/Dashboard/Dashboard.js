import React, { useEffect } from 'react'
import styled from 'styled-components'
import { InnerLayout } from '../../styles/layouts';
import Chart from '../Chart/Chart';
import { dollar } from '../../utils/Icons';
import { useGlobalContext } from '../../context/globalContext';

function Dashboard() {

  const {getExpenses, getIncomes, totalExpense, totalIncome, totalBalance, incomes, expenses, checkName, getName} = useGlobalContext()

  useEffect(() => {
    getIncomes()
    getExpenses()
    checkName()
    getName()
  }, [])

  return (
    <DashboardStyled>
        <InnerLayout>
            <div className="stats-con">
              <div className="chart-con">
                <Chart />
              </div>
              <div className="history-con">
                <h2 className="salary-title">Min <span>Salary</span>Max</h2>
                <div className="salary-item">
                  <p>
                    ${Math.min(...incomes.map(item => item.amount))}
                  </p>
                  <p>
                    ${Math.max(...incomes.map(item => item.amount))}
                  </p>
                </div>
                <h2 className="salary-title">Min <span>Expense</span>Max</h2>
                <div className="salary-item">
                  <p>
                    ${Math.min(...expenses.map(item => item.amount))}
                  </p>
                  <p>
                    ${Math.max(...expenses.map(item => item.amount))}
                  </p>
                </div>
              </div>
            </div>
            <div className="amount-con">
              <div className="income">
                <h2>Total Income</h2>
                <p>
                  {dollar} {totalIncome()}
                </p>
              </div>
              <div className="expense">
                <h2>Total Expense</h2>
                <p>
                  {dollar} {totalExpense()}
                </p>
              </div>
              <div className="balance">
                <h2>Total Balance</h2>
                <p>
                  {dollar} {totalBalance()}
                </p>
              </div>
            </div>
        </InnerLayout>
    </DashboardStyled>
  )
}

const DashboardStyled = styled.div`
    .stats-con{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 2rem;
        .chart-con{
            grid-column: 1 / 4;
            height: 400px;
        }

        .history-con{
            grid-column: 4 / -1;
            h2{
                margin: 1rem 0;
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .salary-title{
                font-size: 1.2rem;
                span{
                    font-size: 1.8rem;
                }
            }
            .salary-item{
                background: #FCF6F9;
                border: 2px solid #FFFFFF;
                box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
                padding: 1rem;
                border-radius: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                p{
                    font-weight: 600;
                    font-size: 1.6rem;
                }
            }
        }
    }

    .amount-con{
        width: 100%;
        display: flex;
        gap: 2rem;
        margin-top: 2rem;
        .income, .expense, .balance{
            flex: 1;
            background: #FCF6F9;
            border: 2px solid #FFFFFF;
            box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
            justify-content: center;
            align-items: center;
            border-radius: 20px;
            padding: 1rem;
            p{
                color: var(--color-green);
                font-size: 3.5rem;
                font-weight: 700;
            }
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
`;

export default Dashboard