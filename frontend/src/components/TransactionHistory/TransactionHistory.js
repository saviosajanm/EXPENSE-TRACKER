import React, { useEffect } from 'react'
import styled from 'styled-components'
import { InnerLayout } from '../../styles/layouts';
import { useGlobalContext } from '../../context/globalContext';
import History from '../History/History';

function TransactionHistory() {

  const {getExpenses, getIncomes, totalExpense, totalIncome, totalBalance, incomes, expenses} = useGlobalContext()

  useEffect(() => {
    getIncomes()
    getExpenses()
  }, [])

  return (
    <TransactionHistoryStyled>
        <InnerLayout>
            <div className="stats-con">
              <div className="history-con">
                <History />
              </div>
            </div>
        </InnerLayout>
    </TransactionHistoryStyled>
  )
}

const TransactionHistoryStyled = styled.div`
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
`;

export default TransactionHistory