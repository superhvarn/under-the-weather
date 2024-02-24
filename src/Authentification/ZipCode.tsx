import {useState} from "react";

export const ZipCode = (props: any) => {
    const [zipCode, setZipCode] = useState('');

    return (
        <div className={"auth-form-container"}>
            <h1>Enter a Zip Code to get Started!</h1>
            <input type={"number"} placeholder={"01234"} value={zipCode} onChange={(e) => setZipCode(e.target.value)}/>
        </div>
    )
}