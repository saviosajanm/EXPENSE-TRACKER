import React, { useEffect, useState } from 'react';
import { useGlobalContext } from '../../context/globalContext';
import styled from 'styled-components';
import { check, cross } from '../../utils/Icons';

function EditableText() {

    const {getName, updateName, name, checkName} = useGlobalContext()
    
    useEffect(() => {
        const fetchData = async () => {
            try {
                await checkName();
                const data = await getName();
                setText(data.name);
                setOriginalText(data.name);
            } catch (error) {
                // Handle errors here
                console.error('Error fetching data:', error);
            }
        };
        
        fetchData();
    }, [getName]);

    const [isEditing, setIsEditing] = useState(false);
    const [text, setText] = useState(name);
    const [originalText, setOriginalText] = useState(name);
    //console.log(name, "---------------------");

    const handleEditClick = () => {
        setText(name)
        setOriginalText(text);
        setIsEditing(true);
    };

    const handleCancelClick = () => {
        setText(originalText);
        setIsEditing(false);
    };

    const handleInputChange = (e) => {
        console.log(text);
        setText(e.target.value);
    };

    const handleSubmit = async () => {
        try {

            await updateName(text);
            setIsEditing(false);
            console.log('Updated Text:', text);
        } catch (error) {
            console.error('Error updating name:', error);
        }
    };
    

    return (
        <EditableTextStyled>
            {isEditing ? (
                <div>
                <input
                    type="text"
                    value={text}
                    onChange={handleInputChange}
                />
                <button className="yes" onClick={handleSubmit}>{check}</button>
                <button className="no" onClick={handleCancelClick}>{cross}</button>
                </div>
            ) : (
                <div>
                <p style={{
                    "color": "rgba(34, 34, 96, 1)",
                    "fontSize": "1.6rem",
                    "fontWeight": "700"
                    }} 
                    onClick={handleEditClick}>{name}</p>
                </div>
            )}
        </EditableTextStyled>
    );
}

const EditableTextStyled = styled.div`
    display: flex;

    input {
        flex: 1;
        height: 30px;
        width: 57%;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid white;
    }

    .yes {
        flex: 1;
        height: 30px;
        width: 15%;
        border-radius: 5px;
        border: 1px solid white;
        margin-left: 5px;
        cursor: pointer;
        background: green;
        color: white;
    }

    .no {
        flex: 1;
        height: 30px;
        width: 15%;
        border-radius: 5px;
        border: 1px solid white;
        margin-left: 5px;
        cursor: pointer;
        background: red;
        color: white;
    }

    p {
        color: rgba(34, 34, 96, 1);
        font-size: 1.6rem;
        font-weight: 700;
        cursor: pointer;
    }
`;

export default EditableText ;
