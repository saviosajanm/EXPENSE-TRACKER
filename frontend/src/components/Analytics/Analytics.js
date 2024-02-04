import React, { useEffect } from 'react'
import styled from 'styled-components'
import AnalyticsChart from '../Chart/AnalyticsChart';
import { InnerLayout } from '../../styles/layouts';
import { useGlobalContext } from '../../context/globalContext';
import Verdict from '../Verdict/Verdict';

function Analytics() {

    //const {addIncome, incomes, getIncomes, deleteIncome, totalIncome} = useGlobalContext()
  
    useEffect(() => {
      //getIncomes()
    }, [])
  
    //<h2 className="total-income">Total Income: <span>${"Heiiii"}</span></h2>

    return (
      <AnalyticsStyled>
          <InnerLayout>
              <AnalyticsChart />
              <Verdict />
          </InnerLayout>
      </AnalyticsStyled>
    )
  }
  
  const AnalyticsStyled = styled.div`
      display: flex;
      overflow: auto;
      .total-income{
          display: flex;
          justify-content: center;
          align-items: center;
          background: #FCF6F9;
          border: 2px solid #FFFFFF;
          box-shadow: 0px 1px 15px rgba(0, 0, 0, 0.06);
          border-radius: 20px;
          padding: 1rem;
          margin: 1rem 0;
          font-size: 2rem;
          gap: .5rem;
          span{
              font-size: 2.5rem;
              font-weight: 800;
              color: var(--color-green);
          }
      }
      .income-content{
          display: flex;
          gap: 2rem;
          .incomes{
              flex: 1;
          }
      }
  `;
  
  export default Analytics