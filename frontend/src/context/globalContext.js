import React, {useContext, useState} from 'react'
import axios from 'axios'
const BASE_URL = "http://localhost:5000/api/v1/";

const GlobalContext = React.createContext()

export const GlobalProvider = ({children}) => {

    const [incomes, setIncomes] = useState([])
    const [expenses, setExpenses] = useState([])
    const [error, setError] = useState(null)
    const [name, setName] = useState("")
    const [preds, setPreds] = useState(-1)
    const [lastm, setLastm] = useState('')
    const [incLen, setIncLen] = useState(0)
    const [expLen, setExpLen] = useState(0)
    const [verdict, setVerdict] = useState([])
    const [incomeVerdict, setIncomeVerdict] = useState("-")
    const [expenseVerdict, setExpenseVerdict] = useState("-")
    const [balanceVerdict, setBalanceVerdict] = useState("-")

    function sumArray(array) {
        let sum = 0;
        array.forEach(element => {
          sum += element;
        });
        return sum.toFixed(2);
    }

    const addIncome = async (income) => {
        const response = await axios.post(`${BASE_URL}add-income`, income)
            .catch((err) => {
                setError(err.response.data.message)
            })
        getIncomes()
    }

    const getIncomes = async () => {
        const response = await axios.get(`${BASE_URL}get-incomes`)
        setIncomes(response.data)
        setIncLen(response.data.length)
    }

    const deleteIncome = async (id) => {
        const res = await axios.delete(`${BASE_URL}delete-income/${id}`)
        getIncomes()
    }

    const totalIncome = () => {
        let totalIncome = 0;
        incomes.forEach((income) => {
            totalIncome += income.amount
        })

        return totalIncome;
    }

    //calculate expenses

    const addExpense = async (expenses) => {
        const response = await axios.post(`${BASE_URL}add-expense`, expenses)
            .catch((err) => {
                setError(err.response.data.message)
            })
        getExpenses()
    }

    const getExpenses = async () => {
        const response = await axios.get(`${BASE_URL}get-expenses`)
        setExpenses(response.data)
        setExpLen(response.data.length)
        //console.log(response.data)
    }

    const deleteExpense = async (id) => {
        const res = await axios.delete(`${BASE_URL}delete-expense/${id}`)
        getExpenses()
    }

    const totalExpense = () => {
        let totalExpense = 0;
        expenses.forEach((expenses) => {
            totalExpense += expenses.amount
        })

        return totalExpense;
    }

    const totalBalance = () => {
        return totalIncome() - totalExpense();
    }

    const transactionHistory = () => {
        const history = [...incomes, ...expenses]
        history.sort((a, b) => {
            return new Date(b.createdAt) - new Date(a.createdAt)
        })

        return history
    }

    const updateName = async (name) => {
        try {
            const response = await axios.post(`${BASE_URL}change-name`, {"name" : name});
            if (response && response.data && response.data[0] && response.data[0].name) {
                setName(response.data[0].name);
            } else {
                throw new Error('Response data is missing or invalid.');
            }
        } catch (error) {
            console.error('Error updating name:', error);
        }
        getName();
    };

    const getName = async () => {
        const response = await axios.get(`${BASE_URL}get-name`)
        //console.log(response.data.name, "=========++++==========");
        setName(response.data.name)
    }

    const checkName = async () => {
        const response = await axios.get(`${BASE_URL}check-name`)
        getName()
    }

    const getPrediction = async (pred) => {
        const response = await axios.post(`${BASE_URL}prediction`, pred)
            .catch((err) => {
                setError(err.response.data.message)
            })
        console.log(response.data.prediction);
        setPreds(response.data.prediction)
        setLastm(response.data.last_month)
    }

    const getVerdict = async (pred) => {
        const response = await axios.post(`${BASE_URL}verdict`, pred)
            .catch((err) => {
                setError(err.response.data.message)
            })
        console.log(response.data.verdict);
        setVerdict(response.data.verdict)
        if (response.data.verdict === -1) {
            setIncomeVerdict("-")
            setExpenseVerdict("-")
            setBalanceVerdict("-")
        } else {
            setIncomeVerdict(sumArray(response.data.verdict[1]))
            setExpenseVerdict(sumArray(response.data.verdict[0]))
            setBalanceVerdict((sumArray(response.data.verdict[1]) - sumArray(response.data.verdict[0])).toFixed(2))
        }
        
    }
    
    //console.log(totalIncome())

    return (
        <GlobalContext.Provider value={{
            addIncome,
            getIncomes,
            incomes,
            deleteIncome,
            totalIncome,
            addExpense,
            getExpenses,
            expenses,
            deleteExpense,
            totalExpense,
            totalBalance,
            transactionHistory,
            error,
            setError,
            getName,
            updateName,
            checkName,
            name,
            getPrediction,
            preds,
            lastm,
            incLen,
            expLen,
            getVerdict,
            verdict,
            incomeVerdict,
            expenseVerdict,
            balanceVerdict
        }}>
            {children}
        </GlobalContext.Provider>
    )
}

export const useGlobalContext = () =>{
    return useContext(GlobalContext)
}