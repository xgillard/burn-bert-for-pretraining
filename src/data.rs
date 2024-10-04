use std::{fs::File, path::Path};

use derive_builder::Builder;
use parquet::{file::reader::SerializedFileReader, record::{Field, Row}};

use crate::error::Error;

pub fn read_parquet<P: AsRef<Path>>(fname: P) -> Result<Vec<BertTrainingInput>, Error> {
    let file = File::open(fname)?;
    let reader = SerializedFileReader::new(file)?;
    let mut data = vec![];

    for row in reader.into_iter() {
        data.push(row?.try_into()?);
    }
    Ok(data)
}

/// A record for pretraining a bert model
#[derive(Debug, Clone, Builder)]
pub struct BertTrainingInput {
    pub input_tokens:        Vec<i32>,
    pub token_type_ids:      Vec<i32>,
    pub attention_mask:      Vec<i32>,
    pub special_tokens_mask: Vec<i32>,
    pub next_sentence_label: bool,
}

impl TryFrom<Row> for BertTrainingInput {
    type Error = crate::error::Error;

    fn try_from(row: Row) -> Result<Self, Self::Error> {
        let mut builder = BertTrainingInputBuilder::create_empty();

        for (column, field) in row.into_columns() {
            match column.as_str() {
                "input_ids"           => { builder.input_tokens(FieldWrapper::from(&field).try_into()?);},
                "token_type_ids"      => { builder.token_type_ids(FieldWrapper::from(&field).try_into()?);},
                "attention_mask"      => { builder.attention_mask(FieldWrapper::from(&field).try_into()?);},
                "special_tokens_mask" => { builder.special_tokens_mask(FieldWrapper::from(&field).try_into()?);},
                "next_sentence_label" => { builder.next_sentence_label(FieldWrapper::from(&field).try_into()?);},
                _ => {/* ignore */}
            }
        }
        Ok(builder.build()?)
    }
}

#[derive(Debug, Clone)]
pub struct FieldWrapper<'a> {
    field: &'a Field
}

impl <'a> From<&'a Field> for FieldWrapper<'a> {
    fn from(field: &'a Field) -> Self {
        FieldWrapper{field}
    }
}

/// Tries to parse a parquet field to a boolean value
impl TryFrom<FieldWrapper<'_>> for bool {
    type Error = crate::error::Error;

    fn try_from(field: FieldWrapper) -> Result<Self, Self::Error> {
        if let Field::Bool(b) = field.field {
            Ok(*b)
        } else {
            Err(Error::NotABoolean)
        }
    }
}

/// Tries to parse a parquet field to an i32 value
impl TryFrom<FieldWrapper<'_>> for i32 {
    type Error = crate::error::Error;

    fn try_from(field: FieldWrapper) -> Result<Self, Self::Error> {
        match field.field {
            Field::Double(x) => Ok(*x as i32),
            Field::Byte(x)    => Ok(*x as i32),
            Field::Float(x)  => Ok(*x as i32),
            Field::Int(x)    => Ok(*x),
            Field::Long(x)   => Ok(*x as i32),
            Field::UByte(x)   => Ok(*x as i32),
            Field::UInt(x)   => Ok(*x as i32),
            Field::ULong(x)  => Ok(*x as i32),
            _ => Err(Error::NotAnInt)
        }
    }
}


impl <'a, T> TryFrom<FieldWrapper<'a>> for Vec<T> 
where T: TryFrom<FieldWrapper<'a>, Error = crate::error::Error>
{
    type Error = crate::error::Error;

    fn try_from(field: FieldWrapper<'a>) -> Result<Self, Self::Error> {
        if let Field::ListInternal(lst) = field.field {
            let mut data = Vec::with_capacity(lst.len());
            for elt in lst.elements() {
                data.push(FieldWrapper::from(elt).try_into()?);
            }
            Ok(data)
        } else {
            Err(Error::NotAList)
        }
    }
}