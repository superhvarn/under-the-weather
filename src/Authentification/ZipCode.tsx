import { useState } from "react";
import axios from 'axios';

export const ZipCode = (props: any) => {
    const [selectedState, setSelectedState] = useState('');

    const handleSubmit = async (e: any) => {
        e.preventDefault();
        try {
            const response = await fetch("http://127.0.0.1:5000/api/state", {
            method: "POST",
            headers: {
            "Content-Type": "application/json",
            },
            body: JSON.stringify({ selectedState }),
        });
            const data = await response.json();
            console.log(data);
        } catch (error) {
            console.error("Error:", error);
        }
    }

    const handleBack = (e: any) => {
        e.preventDefault();
        props.onFormSwitch('register');
    }

    // List of US states with abbreviations
    const states = [
    { abbreviation: "AL", name: "Alabama" },
    { abbreviation: "AK", name: "Alaska" },
    { abbreviation: "AZ", name: "Arizona" },
    { abbreviation: "AR", name: "Arkansas" },
    { abbreviation: "CA", name: "California" },
    { abbreviation: "CO", name: "Colorado" },
    { abbreviation: "CT", name: "Connecticut" },
    { abbreviation: "DE", name: "Delaware" },
    { abbreviation: "FL", name: "Florida" },
    { abbreviation: "GA", name: "Georgia" },
    { abbreviation: "HI", name: "Hawaii" },
    { abbreviation: "ID", name: "Idaho" },
    { abbreviation: "IL", name: "Illinois" },
    { abbreviation: "IN", name: "Indiana" },
    { abbreviation: "IA", name: "Iowa" },
    { abbreviation: "KS", name: "Kansas" },
    { abbreviation: "KY", name: "Kentucky" },
    { abbreviation: "LA", name: "Louisiana" },
    { abbreviation: "ME", name: "Maine" },
    { abbreviation: "MD", name: "Maryland" },
    { abbreviation: "MA", name: "Massachusetts" },
    { abbreviation: "MI", name: "Michigan" },
    { abbreviation: "MN", name: "Minnesota" },
    { abbreviation: "MS", name: "Mississippi" },
    { abbreviation: "MO", name: "Missouri" },
    { abbreviation: "MT", name: "Montana" },
    { abbreviation: "NE", name: "Nebraska" },
    { abbreviation: "NV", name: "Nevada" },
    { abbreviation: "NH", name: "New Hampshire" },
    { abbreviation: "NJ", name: "New Jersey" },
    { abbreviation: "NM", name: "New Mexico" },
    { abbreviation: "NY", name: "New York" },
    { abbreviation: "NC", name: "North Carolina" },
    { abbreviation: "ND", name: "North Dakota" },
    { abbreviation: "OH", name: "Ohio" },
    { abbreviation: "OK", name: "Oklahoma" },
    { abbreviation: "OR", name: "Oregon" },
    { abbreviation: "PA", name: "Pennsylvania" },
    { abbreviation: "RI", name: "Rhode Island" },
    { abbreviation: "SC", name: "South Carolina" },
    { abbreviation: "SD", name: "South Dakota" },
    { abbreviation: "TN", name: "Tennessee" },
    { abbreviation: "TX", name: "Texas" },
    { abbreviation: "UT", name: "Utah" },
    { abbreviation: "VT", name: "Vermont" },
    { abbreviation: "VA", name: "Virginia" },
    { abbreviation: "WA", name: "Washington" },
    { abbreviation: "WV", name: "West Virginia" },
    { abbreviation: "WI", name: "Wisconsin" },
    { abbreviation: "WY", name: "Wyoming" }
];

    return (
        <div className={"auth-form-container"}>
            <h1>Enter a State to get Started!</h1>
            <select value={selectedState} onChange={(e) => setSelectedState(e.target.value)}>
                <option value="">Select a state</option>
                {states.map((state, index) => (
                    <option key={index} value={state.abbreviation}>{state.name}</option>
                ))}
            </select>
            <button type={"submit"} onClick={handleSubmit}>Submit</button>
            <button onClick={handleBack}>Back</button>
        </div>
    );
};
 