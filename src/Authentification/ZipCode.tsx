import {useState} from "react";

export const ZipCode = (props: any) => {
    const [zipCode, setZipCode] = useState('');

    const handleSubmit = async (e: any) => {
        e.preventDefault();
        console.log("Submit button pressed")
    }

    return (
        <div className={"auth-form-container"}>
            <h1>Enter a Zip Code to get Started!</h1>
            <input type={"text"} placeholder={"01234"} value={zipCode} onChange={(e) => setZipCode(e.target.value)}/>
            <button type={"submit"} onClick={handleSubmit}>Submit</button>
        </div>
    )
}